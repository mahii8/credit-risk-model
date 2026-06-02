from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Credit risk scoring model for buy-now-pay-later service",
    version="1.0.0"
)

# Global model variable
model = None

def load_model():
    """Load or train the model"""
    global model
    try:
        # Try loading saved model
        model_path = "data/processed/rf_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded from disk")
        else:
            # Train fresh model
            print("Training model...")
            df = pd.read_csv("data/processed/customer_features_labeled.csv")
            df = df.dropna(subset=['is_high_risk'])

            feature_cols = [
                'Recency','Frequency','Monetary','Avg_Amount',
                'Std_Amount','Max_Amount','Min_Amount','Total_Value',
                'Avg_Value','Total_Fraud','Fraud_Rate','Unique_Products',
                'Unique_Channels','Unique_Categories','Avg_Hour',
                'Avg_DayOfWeek','Weekend_Ratio'
            ]
            X = df[feature_cols]
            y = df['is_high_risk']

            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model = RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=42, class_weight='balanced'
            )
            model.fit(X_train, y_train)

            # Save model
            os.makedirs("data/processed", exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print("Model trained and saved")
    except Exception as e:
        print(f"Model loading error: {e}")


def risk_to_credit_score(probability: float) -> int:
    """
    Convert risk probability to credit score (300-850 scale).
    Higher score = lower risk = better creditworthiness.
    """
    score = int(850 - (probability * 550))
    return max(300, min(850, score))


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/")
def root():
    return {
        "message": "Bati Bank Credit Risk API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """
    Predict credit risk probability for a customer.

    Returns:
    - risk_probability: 0.0 (low risk) to 1.0 (high risk)
    - risk_label: Low Risk or High Risk
    - credit_score: 300 (worst) to 850 (best)
    - recommendation: Loan decision recommendation
    """
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded")
    try:
        features = pd.DataFrame([customer.dict()])
        prob = model.predict_proba(features)[0][1]
        label = "High Risk" if prob >= 0.5 else "Low Risk"
        score = risk_to_credit_score(prob)

        if prob < 0.3:
            rec = "APPROVE — Low risk customer. Eligible for full credit limit."
        elif prob < 0.5:
            rec = "APPROVE with conditions — Moderate risk. Recommend reduced credit limit."
        elif prob < 0.7:
            rec = "REVIEW — Elevated risk. Manual underwriting review recommended."
        else:
            rec = "DECLINE — High risk customer. Does not meet creditworthiness threshold."

        return PredictionResponse(
            customer_risk_probability=round(float(prob), 4),
            risk_label=label,
            credit_score=score,
            recommendation=rec
        )
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Prediction error: {str(e)}")