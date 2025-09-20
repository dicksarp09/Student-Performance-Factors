import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from src.pipelines.inference_pipeline import predict

# rest of your code...


from flask import Flask, request, jsonify
from src.pipelines.inference_pipeline import predict

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to Student Performance Predictor API",
        "available_routes": {
            "POST /predict/linear": "Predict exam score using Linear Regression",
            "POST /predict/poly": "Predict exam score using Polynomial Regression"
        }
    })

@app.route("/predict/linear", methods=["POST"])
def linear():
    try:
        data = request.json
        pred = predict("student_performance_linear_pipeline.pkl", data)
        return jsonify({"model": "Linear Regression", "prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict/poly", methods=["POST"])
def poly():
    try:
        data = request.json
        pred = predict("student_performance_poly_pipeline.pkl", data)
        return jsonify({"model": "Polynomial Regression", "prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
