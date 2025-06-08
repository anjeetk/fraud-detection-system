import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Preprocess transaction data"""
    # Numeric columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = pd.get_dummies(df, columns=cat_cols)
    
    # Feature engineering
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
    
    return df

def align_features(df, reference_columns):
    """Ensure feature alignment with training data"""
    missing_cols = set(reference_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    return df[reference_columns]