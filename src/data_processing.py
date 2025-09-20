# src/data_processing.py
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def fill_missing(df):
    return df.fillna(df.mode().iloc[0])

def remove_outliers(df, numeric_features):
    df_clean = df.copy()
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean
