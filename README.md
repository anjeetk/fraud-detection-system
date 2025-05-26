# Fraud Detection System Prototype
A fraud detection system which is used to track whether a transaction is fraud or not using ml algorithms
A hybrid machine learning system for real-time fraud detection using:
- XGBoost for supervised learning
- Isolation Forest for anomaly detection
- Dash for interactive dashboard

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Running the Dashboard](#running-the-dashboard)
6. [Folder Structure](#folder-structure)

## Prerequisites
- Python 3.8-3.11
- IEEE-CIS Fraud Detection Dataset (place in `data/` folder)
- 8GB+ RAM recommended for full dataset training

## Installation

# Clone repository
git clone https://github.com/anjeetk/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

## Data Preparation
- Download IEEE-CIS dataset

- Place these files in data/ folder:

    1. train_transaction.csv
    2. train_identity.csv
    3. test_transaction.csv
    4. test_identity.csv


## model training 
python train.py

## Running the Dashboard      
python dashboard.py

## Folder Structure
fraud-detection/
├── data/               # Raw dataset files (ignored in git)
├── models/             # Saved models (ignored)
├── outputs/            # Generated reports (ignored)
├── utils/              # Helper functions
│   └── helpers.py
├── .gitignore
├── config.py           # Model parameters
├── dashboard.py        # Interactive UI
├── predict.py          # Prediction functions
├── README.md           # This file
├── requirements.txt    # Dependencies
└── train.py            # Model training
