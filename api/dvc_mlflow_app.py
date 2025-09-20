import os
import joblib
import pandas as pd
import mlflow
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

linear_model = joblib.load(os.path.join(MODEL_DIR, "linear_model.pkl"))
poly_model = joblib.load(os.path.join(MODEL_DIR, "poly_model.pkl"))


# Configure MLflow experiment
mlflow.set_tracking_uri("http://localhost:5002") 
mlflow.set_experiment("Student_Performance_API")


@app.route("/", methods=["GET"])
def home():
    return {
        "message": "Welcome to the Student Performance Prediction API!",
        "usage": "Send a POST request to /predict with JSON input"
    }



@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features_dict = data["features"]

    features_df = pd.DataFrame([features_dict])

    
    prediction_linear = linear_model.predict(features_df)[0]
    prediction_poly = poly_model.predict(features_df)[0]

    
    mlflow.log_params(features_dict)  # log all input features
    mlflow.log_metric("linear_prediction", float(prediction_linear))
    mlflow.log_metric("poly_prediction", float(prediction_poly))

    return jsonify({
        "linear_prediction": float(prediction_linear),
        "polynomial_prediction": float(prediction_poly)
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
