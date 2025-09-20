import pandas as pd
from src.pipelines.model_service import load_model

def predict(model_name, input_dict):
    # Load model from disk
    model = load_model(model_name)

    # Wrap input into a DataFrame (pipeline expects this format)
    df = pd.DataFrame([input_dict])

    # Run prediction
    prediction = model.predict(df)[0]

    return float(prediction)
