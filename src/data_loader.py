# src/data_loader.py

import pandas as pd

def load_fraud_data(fraud_path: str) -> pd.DataFrame:
    """Load Fraud_Data.csv (e-commerce)"""
    return pd.read_csv(fraud_path)

def load_ip_country_data(ip_path: str) -> pd.DataFrame:
    """Load IpAddress_to_Country.csv"""
    return pd.read_csv(ip_path)

def load_credit_card_data(credit_path: str) -> pd.DataFrame:
    """Load creditcard.csv (bank)"""
    return pd.read_csv(credit_path)
