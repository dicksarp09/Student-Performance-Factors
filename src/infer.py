import joblib
import pandas as pd

def load_model(model_path: str):
    return joblib.load(model_path)

def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])  # convert dict → DataFrame
    prediction = model.predict(df)
    return float(prediction[0])
