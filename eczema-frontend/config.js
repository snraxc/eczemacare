const CONFIG = {
    API_BASE_URL: (
        window.location.protocol === 'file:' || 
        window.location.hostname === 'localhost' || 
        window.location.hostname === '127.0.0.1'
    )
        ? 'http://127.0.0.1:8000'                          // local
        : 'https://eczemacare1.onrender.com'                // production (Render)
};
