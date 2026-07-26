import os
import json
import uuid
import shutil
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from PIL import Image
import io

load_dotenv()

# ==========================================
# 1. Configuration & Setup
# ==========================================
SECRET_KEY = "SdSdsfS#dfdfdfdfsdfdfddfdfdfdf#4dfdffgd"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days

# Email (SMTP) settings for sending verification codes.
# Works with any SMTP provider (Gmail, Outlook, SendGrid, Mailgun, etc.) -
# just set these env vars (e.g. in a .env file, see .env.example).
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
FROM_EMAIL = os.environ.get("FROM_EMAIL", SMTP_USER)
FROM_NAME = os.environ.get("FROM_NAME", "EczemaCare")
VERIFICATION_CODE_EXPIRE_MINUTES = 10

#app = FastAPI(title="EczemaCare API")
app = FastAPI(title="EczemaCare API", root_path="/eczema-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Create directories to save images and serve them
# ==========================================
os.makedirs("uploads/original", exist_ok=True)
os.makedirs("uploads/cropped", exist_ok=True)

# Create a directory specifically for the database
os.makedirs("data", exist_ok=True)

# Load ML Model
print("Loading ML Model...")
model = tf.keras.models.load_model('eczema_model.h5')
IMG_SIZE = (128, 128)

# ==========================================
# Probability calibration & threshold tuning
# ==========================================
# The raw model output is a single sigmoid probability. Rather than always cutting
# it at 0.5, we calibrate it (Platt scaling) against the validation set and tune the
# decision threshold + an "uncertain" band in that calibrated space. See /admin/recalibrate.
CALIBRATION_PATH = "model_calibration.json"
DEFAULT_CALIBRATION = {"calibration_A": 1.0, "calibration_B": 0.0, "tuned_threshold": 0.5, "uncertain_margin": 0.0}

def load_calibration():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            data = json.load(f)
        return {
            "calibration_A": data.get("calibration_A", 1.0),
            "calibration_B": data.get("calibration_B", 0.0),
            "tuned_threshold": data.get("tuned_threshold", 0.5),
            "uncertain_margin": data.get("uncertain_margin", 0.0),
        }
    return dict(DEFAULT_CALIBRATION)

CALIBRATION = load_calibration()

def calibrate_probability(raw_prob: float) -> float:
    z = CALIBRATION["calibration_A"] * raw_prob + CALIBRATION["calibration_B"]
    return float(1 / (1 + np.exp(-z)))

# ==========================================
# Bayesian prior adjustment
# ==========================================
# The calibration above was fit on eczema_dataset/val, which is an artificially
# balanced 50/50 split of normal vs eczema images. That balance is very unlikely to
# match the real prevalence of eczema among photos actual users submit. Bayes'
# theorem lets us correct for that mismatch explicitly:
#
#   P(Eczema | image) = P(image | Eczema) * P(Eczema)
#                        --------------------------------
#                        P(image | Eczema) * P(Eczema) + P(image | Normal) * P(Normal)
#
# calibrated_prob already IS P(Eczema | image) under the assumption P(Eczema) = 0.5,
# which means the likelihood ratio P(image|Eczema)/P(image|Normal) is recoverable as
# calibrated_prob / (1 - calibrated_prob). Plugging that ratio back into Bayes' rule
# with a different prior gives the formula below. Set ECZEMA_PRIOR in .env to your
# best estimate of real-world prevalence among your users (0 < prior < 1); it
# defaults to 0.5, which leaves predictions unchanged (no adjustment).
ECZEMA_PRIOR = float(os.environ.get("ECZEMA_PRIOR", "0.5"))

def apply_bayes_prior(calibrated_prob: float, prior: float = None) -> float:
    if prior is None:
        prior = ECZEMA_PRIOR
    numerator = calibrated_prob * prior
    denominator = numerator + (1 - calibrated_prob) * (1 - prior)
    if denominator <= 0:
        return calibrated_prob
    return numerator / denominator

# ==========================================
# 2. Database Setup (SQLAlchemy)
# Can apply to Render web service
# ==========================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/eczema_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    display_name = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)
    verification_code = Column(String, nullable=True)
    verification_expires = Column(DateTime, nullable=True)

# ==========================================
# Updated Database Schema (Added image_path)
# ==========================================
class Record(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_label = Column(String)
    confidence = Column(Float)
    image_path = Column(String, nullable=True) # <-- NEW: Save the cropped image path

Base.metadata.create_all(bind=engine)

# Lightweight migration: the users table may already exist on disk from before
# email verification was added, so make sure the new columns are present.
def _ensure_column(table: str, column: str, coltype: str):
    with engine.connect() as conn:
        existing_cols = [row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))]
        if column not in existing_cols:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))
            conn.commit()

_ensure_column("users", "is_verified", "BOOLEAN DEFAULT 0")
_ensure_column("users", "verification_code", "VARCHAR")
_ensure_column("users", "verification_expires", "DATETIME")
_ensure_column("users", "display_name", "VARCHAR")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 3. Authentication & Security
# ==========================================
import bcrypt

def get_password_hash(password: str) -> str:
    # Hash a password for the first time
    # (Using bcrypt, the salt is saved into the hash itself)
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Check a hashed password against the provided plain password
    return bcrypt.checkpw(
        plain_password.encode('utf-8'), 
        hashed_password.encode('utf-8')
    )

def generate_verification_code() -> str:
    return "".join(secrets.choice("0123456789") for _ in range(6))

def send_verification_email(to_email: str, code: str):
    if not SMTP_USER or not SMTP_PASSWORD:
        # Dev fallback: no SMTP credentials configured yet, so just log the
        # code instead of failing. Set SMTP_USER/SMTP_PASSWORD (see .env.example)
        # to actually send real emails.
        print(f"[DEV] No SMTP configured - verification code for {to_email}: {code}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your EczemaCare verification code"
    msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
    msg["To"] = to_email

    text_body = (
        f"Your EczemaCare verification code is: {code}\n\n"
        f"This code expires in {VERIFICATION_CODE_EXPIRE_MINUTES} minutes."
    )
    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: auto;">
        <h2 style="color:#2563eb;">EczemaCare</h2>
        <p>Your verification code is:</p>
        <p style="font-size: 32px; font-weight: bold; letter-spacing: 6px;">{code}</p>
        <p style="color:#6b7280; font-size: 13px;">
            This code expires in {VERIFICATION_CODE_EXPIRE_MINUTES} minutes.
            If you didn't request this, you can ignore this email.
        </p>
    </div>
    """
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# Add this new dependency function for optional users
# Set auto_error=False so guests can access endpoints without a token throwing a 401 error
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
def get_optional_user(token: str = Depends(oauth2_scheme_optional), db: Session = Depends(get_db)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    return db.query(User).filter(User.username == username).first()

# ==========================================
# 4. API Endpoints
# ==========================================
class VerifyEmailRequest(BaseModel):
    email: str
    code: str

class ResendCodeRequest(BaseModel):
    email: str

@app.post("/register")
def register(
    user_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    db_user = db.query(User).filter(User.username == user_data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_pw = get_password_hash(user_data.password)
    code = generate_verification_code()
    new_user = User(
        username=user_data.username,
        display_name=None,
        hashed_password=hashed_pw,
        is_verified=False,
        verification_code=code,
        verification_expires=datetime.utcnow() + timedelta(minutes=VERIFICATION_CODE_EXPIRE_MINUTES),
    )
    db.add(new_user)
    db.commit()
    send_verification_email(new_user.username, code)
    return {"message": "Verification code sent to your email"}

@app.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "email": current_user.username,
        "username": current_user.display_name or current_user.username,
        "needs_username": not current_user.display_name,
    }

class SetUsernameRequest(BaseModel):
    display_name: str

@app.post("/set-username")
def set_username(
    payload: SetUsernameRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = payload.display_name.strip()
    if len(name) < 2 or len(name) > 30:
        raise HTTPException(status_code=400, detail="Username must be between 2 and 30 characters")
    current_user.display_name = name
    db.commit()
    return {"message": "Username set successfully", "username": current_user.display_name}

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

@app.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    current_user.hashed_password = get_password_hash(payload.new_password)
    db.commit()
    return {"message": "Password updated successfully"}

@app.post("/verify-email")
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="No account found for that email")
    if user.is_verified:
        return {"message": "Email already verified"}
    if not user.verification_code or user.verification_code != payload.code:
        raise HTTPException(status_code=400, detail="Incorrect verification code")
    if not user.verification_expires or user.verification_expires < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Verification code expired, please request a new one")

    user.is_verified = True
    user.verification_code = None
    user.verification_expires = None
    db.commit()

    access_token = create_access_token(data={"sub": user.username})
    return {"message": "Email verified", "access_token": access_token, "token_type": "bearer"}

@app.post("/resend-code")
def resend_code(payload: ResendCodeRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="No account found for that email")
    if user.is_verified:
        return {"message": "Email already verified"}

    code = generate_verification_code()
    user.verification_code = code
    user.verification_expires = datetime.utcnow() + timedelta(minutes=VERIFICATION_CODE_EXPIRE_MINUTES)
    db.commit()
    send_verification_email(user.username, code)
    return {"message": "Verification code resent"}

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Email not verified")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# ==========================================
# Recalibration: re-derive the decision threshold, Platt-scaling calibration,
# and "uncertain" band from the validation set. Run this after retraining the
# model, or any time you want to refresh these numbers.
# ==========================================
@app.post("/admin/recalibrate")
def recalibrate():
    from sklearn.linear_model import LogisticRegression

    val_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "eczema_dataset", "val"))
    raw_probs = []
    true_labels = []

    for label_name, true_label in [("normal", 0), ("eczema", 1)]:
        folder = os.path.join(val_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = Image.open(os.path.join(folder, fname)).convert("RGB").resize(IMG_SIZE)
            arr = np.expand_dims(np.array(img) / 255.0, axis=0)
            raw_probs.append(float(model.predict(arr, verbose=0)[0][0]))
            true_labels.append(true_label)

    if len(raw_probs) < 10:
        raise HTTPException(status_code=400, detail=f"Only found {len(raw_probs)} validation images at {val_dir}; need at least 10 to recalibrate")

    raw_probs = np.array(raw_probs)
    true_labels = np.array(true_labels)

    baseline_acc = float(((raw_probs > 0.5).astype(int) == true_labels).mean())

    # Platt scaling: fit calibrated_prob = sigmoid(A * raw_prob + B) against true labels
    lr = LogisticRegression()
    lr.fit(raw_probs.reshape(-1, 1), true_labels)
    calib_A = float(lr.coef_[0][0])
    calib_B = float(lr.intercept_[0])
    calibrated_probs = 1 / (1 + np.exp(-(calib_A * raw_probs + calib_B)))

    # Threshold tuning: sweep cutoffs in calibrated-probability space, maximize accuracy
    best_threshold, best_acc = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        acc = float(((calibrated_probs > t).astype(int) == true_labels).mean())
        if acc > best_acc:
            best_acc, best_threshold = acc, float(t)

    # Uncertain band: flag roughly the least-confident 10% of validation predictions
    # (bottom decile of distance-to-threshold). We deliberately do NOT size this off
    # the validation errors themselves - with only ~40 validation images there are
    # just a handful of errors, and fitting the margin to cover them is unstable
    # (an earlier version of this flagged 67% of all predictions as "Uncertain").
    errors_mask = (calibrated_probs > best_threshold).astype(int) != true_labels
    distances = np.abs(calibrated_probs - best_threshold)
    margin = float(np.percentile(distances, 10))
    margin = min(max(margin, 0.03), 0.25)

    uncertain_mask = distances < margin
    confident_mask = ~uncertain_mask
    if confident_mask.any():
        confident_preds = (calibrated_probs[confident_mask] > best_threshold).astype(int)
        accuracy_excluding_uncertain = float((confident_preds == true_labels[confident_mask]).mean())
    else:
        accuracy_excluding_uncertain = None

    result = {
        "n_validation_images": len(raw_probs),
        "baseline_threshold": 0.5,
        "baseline_accuracy": baseline_acc,
        "tuned_threshold": best_threshold,
        "tuned_accuracy": best_acc,
        "calibration_A": calib_A,
        "calibration_B": calib_B,
        "uncertain_margin": margin,
        "n_errors_caught_by_uncertain_band": int((errors_mask & uncertain_mask).sum()),
        "n_total_errors": int(errors_mask.sum()),
        "fraction_flagged_uncertain": float(uncertain_mask.mean()),
        "accuracy_on_confident_predictions": accuracy_excluding_uncertain,
    }

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(result, f, indent=2)

    global CALIBRATION
    CALIBRATION = load_calibration()

    return result

# ==========================================
# Updated Predict Endpoint (Accepts two files)
# ==========================================
@app.post("/predict")
async def predict_and_save(
    original_file: UploadFile = File(...), # <-- Receive original image
    cropped_file: UploadFile = File(...),  # <-- Receive cropped image
    current_user: User = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    # 1. Generate unique filenames using UUID to prevent naming conflicts
    unique_id = str(uuid.uuid4())
    orig_path = f"uploads/original/{unique_id}.jpg"
    crop_path = f"uploads/cropped/{unique_id}.jpg"

    # 2. Save Original Image to disk
    with open(orig_path, "wb") as buffer:
        shutil.copyfileobj(original_file.file, buffer)

    # 3. Read and Save Cropped Image to disk
    crop_contents = await cropped_file.read()
    with open(crop_path, "wb") as buffer:
        buffer.write(crop_contents)

    # 4. Image Preprocessing for Model
    image = Image.open(io.BytesIO(crop_contents)).convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Model Prediction
    prediction = model.predict(img_array)
    raw_prob = float(prediction[0][0])
    calibrated_prob = calibrate_probability(raw_prob)
    bayes_prob = apply_bayes_prior(calibrated_prob)

    threshold = CALIBRATION["tuned_threshold"]
    margin = CALIBRATION["uncertain_margin"]

    # Confidence always means "how strongly bayes_prob leans toward the side it's
    # on" - including for "Uncertain", so the number means the same thing across all
    # three labels (higher = more confident that direction, just not confident enough
    # to cross the uncertain band around the threshold).
    if abs(bayes_prob - threshold) < margin:
        label = "Uncertain"
    elif bayes_prob > threshold:
        label = "Eczema"
    else:
        label = "Normal"
    confidence = bayes_prob * 100 if bayes_prob > threshold else (1 - bayes_prob) * 100
    
    # 6. Save to Database ONLY if the user is logged in
    if current_user:
        # Save the relative path so the frontend can request it
        relative_image_url = f"/{crop_path}" 
        new_record = Record(
            user_id=current_user.id, 
            prediction_label=label, 
            confidence=confidence,
            image_path=relative_image_url # <-- Save path here
        )
        db.add(new_record)
        db.commit()
        print(f"Record and images saved for user: {current_user.username}")
    
    return {"label": label, "confidence": f"{confidence:.2f}%"}

@app.get("/uploads/{folder}/{filename}")
def serve_image(folder: str, filename: str):
    #file_path = f"/app/uploads/{folder}/{filename}"
    import os
    file_path = os.path.join("uploads", folder, filename)

    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"detail": "Image strictly not found on disk"}

# ==========================================
# Updated History Endpoint (Returns image_path)
# ==========================================
@app.get("/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    records = db.query(Record).filter(Record.user_id == current_user.id).order_by(Record.timestamp.desc()).all()
    return [
        {
            # Format as ISO 8601 with 'Z' to indicate it's strictly UTC time
            "date": r.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), 
            "label": r.prediction_label, 
            "confidence": f"{r.confidence:.2f}%",
            "image_path": r.image_path
        } 
        for r in records
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)