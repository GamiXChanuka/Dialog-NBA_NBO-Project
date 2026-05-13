import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the 'ML model' directory to the Python path so we can import main.py
ml_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ML model'))
if ml_model_dir not in sys.path:
    sys.path.append(ml_model_dir)

import main as ml_main

app = FastAPI(
    title="Dialog NBA/NBO Prediction API",
    description="REST API to predict the best action and best offer for customers.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing) so your UI can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the expected input payload based on your ML model's features
class CustomerInput(BaseModel):
    district: str = Field(default="Colombo", description="District (e.g. Colombo, Kandy, Galle)")
    age_group: str = Field(default="26-35", description="Age group (18-25/26-35/36-45/46-60/60+)")
    current_package: str = Field(default="20GB", description="Current package (10GB/20GB/50GB/100GB)")
    monthly_data_usage_gb: float = Field(default=35.0, description="Monthly data usage (GB)")
    youtube_usage_gb: float = Field(default=10.0, description="YouTube usage (GB)")
    social_usage_gb: float = Field(default=8.0, description="Social media usage (GB)")
    voice_minutes: int = Field(default=500, description="Voice minutes / month")
    sms_usage: int = Field(default=100, description="SMS count / month")
    monthly_spend_lkr: int = Field(default=5000, description="Monthly spend (LKR)")
    reload_frequency: int = Field(default=10, description="Reload frequency (times/month)")
    add_on_count: int = Field(default=2, description="Active add-ons count")
    device_type: str = Field(default="4G_Mobile", description="Device type (Basic_Phone/4G_Mobile/5G_Mobile)")
    is_5g_supported: int = Field(default=0, description="Device 5G supported? (1=Yes, 0=No)")
    sim_type: str = Field(default="4G", description="SIM type (4G/5G)")
    router_owned: int = Field(default=0, description="Owns router? (1=Yes, 0=No)")
    churn_risk_score: float = Field(default=0.3, description="Churn risk score (0.0–1.0)")
    complaint_count: int = Field(default=1, description="Complaint count")
    previous_offer_response: str = Field(default="Ignored", description="Previous offer response (Accepted/Ignored/Rejected)")
    days_since_last_change: int = Field(default=180, description="Days since last plan change")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Dialog NBA/NBO Prediction API. Go to /docs for the interactive API documentation."}


@app.post("/predict")
def predict_nba_nbo(customer: CustomerInput):
    """
    Endpoint to predict the Best Action and Best Offer based on customer features.
    """
    try:
        # Convert the Pydantic model to a standard dictionary
        user_input_dict = customer.model_dump()
        
        # Call the existing predict function from main.py
        prediction_results = ml_main.predict(user_input_dict)
        
        return {
            "status": "success",
            "data": prediction_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # This block allows you to run `python app.py` directly to start the server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
