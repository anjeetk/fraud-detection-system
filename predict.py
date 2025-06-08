import pandas as pd
import numpy as np
import joblib
from config import *
from utils.helpers import preprocess_data, align_features

def load_model():
    """Load trained model components"""
    model = joblib.load(MODEL_PATH)
    return (
        model['xgb_model'],
        model['iso_forest'],
        model['scaler'],
        model['feature_columns']
    )


def predict_transaction(transaction_data):
    """Wrapper for dashboard compatibility"""
    xgb_model, iso_forest, scaler, feature_columns = load_model()
    transaction_df = pd.DataFrame([transaction_data])
    transaction_df = preprocess_data(transaction_df)
    transaction_df = align_features(transaction_df, feature_columns)
    transaction_scaled = scaler.transform(transaction_df)
    
    xgb_score = xgb_model.predict_proba(transaction_scaled)[0, 1]
    iso_score = -iso_forest.score_samples(transaction_scaled)[0]
    combined_score = 0.7 * xgb_score + 0.3 * iso_score
    
    return {
        'is_fraud': combined_score > 0.5,
        'confidence': combined_score,
        'xgb_score': xgb_score,
        'anomaly_score': iso_score
    }

# Example usage
if __name__ == "__main__":
    example_transaction = {
        'TransactionAmt': 250.00,
        'card1': 12345,
        'card2': 67890,
        'card4': 'visa',
        'P_emaildomain': 'gmail.com',
        # Add all other required features
    }
    
    result = predict_transaction(example_transaction)
    print("\nFraud Prediction Result:")
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"XGBoost Score: {result['xgb_score']:.4f}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")