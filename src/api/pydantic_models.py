from pydantic import BaseModel
from typing import Optional


class CustomerFeatures(BaseModel):
    """Input features for credit risk prediction"""
    Recency: float
    Frequency: float
    Monetary: float
    Avg_Amount: float
    Std_Amount: float
    Max_Amount: float
    Min_Amount: float
    Total_Value: float
    Avg_Value: float
    Total_Fraud: float
    Fraud_Rate: float
    Unique_Products: float
    Unique_Channels: float
    Unique_Categories: float
    Avg_Hour: float
    Avg_DayOfWeek: float
    Weekend_Ratio: float

    class Config:
        json_schema_extra = {
            "example": {
                "Recency": 30,
                "Frequency": 5,
                "Monetary": 50000,
                "Avg_Amount": 10000,
                "Std_Amount": 5000,
                "Max_Amount": 25000,
                "Min_Amount": 1000,
                "Total_Value": 50000,
                "Avg_Value": 10000,
                "Total_Fraud": 0,
                "Fraud_Rate": 0.0,
                "Unique_Products": 3,
                "Unique_Channels": 2,
                "Unique_Categories": 2,
                "Avg_Hour": 12.5,
                "Avg_DayOfWeek": 2.5,
                "Weekend_Ratio": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Output from credit risk prediction"""
    customer_risk_probability: float
    risk_label: str
    credit_score: int
    recommendation: str