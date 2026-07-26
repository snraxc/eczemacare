pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

********System Design Document********
1. Project Overview
Name: EczemaCare Tracker
Objective: To provide a secure, personalized web platform where users can log in, upload photos of their skin, crop the affected areas, and use a trained AI model (TensorFlow/Keras) to classify the image (Eczema vs. Normal) and track their symptoms over time.

2. System Architecture & Tech Stack
Frontend (Client): * HTML5, CSS3 (Tailwind CSS for rapid UI styling), Vanilla JavaScript.
    Cropper.js (for client-side image cropping).
    Chart.js (optional, for visualizing confidence trends over time).

Backend (API Server): * FastAPI (Python) for high-performance, asynchronous API routing.
    JWT (JSON Web Tokens) for secure user authentication.
    Machine Learning: * TensorFlow/Keras (MobileNetV2 .h5 model).

Database: * SQLite (via SQLAlchemy ORM) for lightweight, serverless relational data storage.

3. Database Schema
We will use two primary tables:
Users Table:
    id (Primary Key, Integer)
    username (String, Unique)
    hashed_password (String)
Records Table (Symptom History):
    id (Primary Key, Integer)
    user_id (Foreign Key -> Users.id)
    timestamp (DateTime)
    prediction_label (String: "Eczema" or "Normal")
    confidence (Float)

4. Training Data Attribution
The eczema training/validation set (eczema_dataset/) is supplemented with 60 images
from the SCIN (Skin Condition Image Network) dataset, selected for skin-tone
diversity (10 per Fitzpatrick skin type, FST1-FST6), filtered to cases where
"Eczema" is the dermatologist-assigned primary label. Filenames are prefixed
"scin_" for provenance.

SCIN dataset citation (required by its data use license):
Ward A, Li J, Wang J, et al. Creating an Empirical Dermatology Dataset Through
Crowdsourcing With Web Search Advertisements. JAMA Netw Open. 2024;7(11):e2446615.
https://doi.org/10.1001/jamanetworkopen.2024.46615
Dataset: https://github.com/google-research-datasets/scin
License: SCIN Data Use License (CC BY-style, attribution required) - see the
LICENSE file in that repository for full terms.