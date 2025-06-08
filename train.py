import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler  # Add this with other imports
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from config import *
from utils.helpers import preprocess_data

# Create directories if they don't exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH.rsplit('/', 1)[0], exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading and sampling 10% of data...")
train_transaction = pd.read_csv(f'{DATA_PATH}train_transaction.csv').sample(frac=SAMPLE_FRACTION, random_state=42)
train_identity = pd.read_csv(f'{DATA_PATH}train_identity.csv')
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

print("Preprocessing data...")
train = preprocess_data(train)

# Prepare features
features = [col for col in train.columns if col not in ['isFraud', 'TransactionID']]
X = train[features]
y = train['isFraud']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
xgb_model.fit(X_train_scaled, y_train)

print("Training Isolation Forest...")
iso_forest = IsolationForest(**ISOLATION_FOREST_PARAMS)
iso_forest.fit(X_train_scaled)

# Save model components
joblib.dump({
    'xgb_model': xgb_model,
    'iso_forest': iso_forest,
    'scaler': scaler,
    'feature_columns': X_train.columns.tolist()
}, MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")