# Configuration constants
DATA_PATH = 'data/'
MODEL_PATH = 'models/fraud_detection_model.pkl'
OUTPUT_PATH = 'outputs/'

# Model parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,  # Reduced for faster training
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'auc'
}

ISOLATION_FOREST_PARAMS = {
    'n_estimators': 50,  # Reduced for faster training
    'contamination': 0.01,
    'random_state': 42
}

SAMPLE_FRACTION = 0.1  # Using 10% of data