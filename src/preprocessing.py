# src/preprocessing.py

import pandas as pd
import numpy as np
from typing import Optional
def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts signup_time and purchase_time to datetime.
    Creates hour_of_day, day_of_week, and time_since_signup features.
    """
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

    return df


def ip_to_int(ip_string: str) -> int:
    """
    Converts a dot-decimal IP string to an integer.
    Returns None if the input is not a valid IP string.
    """
    try:
        parts = str(ip_string).split('.')
        if len(parts) != 4:
            return None
        return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    except Exception:
        return None


def add_country_column(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps IP addresses in the fraud dataframe to countries using IP range data.
    Adds a new 'ip_integer' and 'country' column to fraud_df.
    """

    # Handle IPs: convert to uint32 if already numeric, otherwise convert from string
    if pd.api.types.is_numeric_dtype(fraud_df['ip_address']):
        fraud_df['ip_integer'] = fraud_df['ip_address'].astype('uint32')
    else:
        fraud_df['ip_integer'] = fraud_df['ip_address'].apply(ip_to_int)

    # Convert IP range boundaries to uint32
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(np.uint32)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(np.uint32)

    # Sort IP ranges for efficient access (optional, can help with future optimization)
    ip_df = ip_df.sort_values('lower_bound_ip_address')

    # Function to find country for one IP integer
    def find_country(ip_int: Optional[int]) -> str:
        if pd.isna(ip_int):
            return 'Unknown'
        match = ip_df[
            (ip_df['lower_bound_ip_address'] <= ip_int) &
            (ip_df['upper_bound_ip_address'] >= ip_int)
        ]
        return match.iloc[0]['country'] if not match.empty else 'Unknown'

    # Apply mapping to get the 'country' column
    fraud_df['country'] = fraud_df['ip_integer'].apply(find_country)

    return fraud_df

