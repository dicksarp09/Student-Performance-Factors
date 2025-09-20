# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os
from pathlib import Path


from data_processing import load_data, fill_missing, remove_outliers
from model_pipeline import get_preprocessor, get_linear_pipeline, get_polynomial_pipeline

# Feature lists
numeric_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
                    'Tutoring_Sessions', 'Physical_Activity']

categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                        'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                        'Parental_Education_Level', 'Distance_from_Home', 'Gender']


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    return mae, mse, rmse, r2

from pathlib import Path
import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split

def main():
    # --- MLflow setup ---
    mlflow.set_tracking_uri("http://host.docker.internal:5002")
    mlflow.set_experiment("student_performance")   

    data_path = Path("data/raw/StudentPerformanceFactors.csv")  
    df = load_data(data_path)
    df = fill_missing(df)
    df = remove_outliers(df, numeric_features)

    X = df[numeric_features + categorical_features]
    y = df['Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = get_preprocessor(numeric_features, categorical_features)

    os.makedirs("models", exist_ok=True)

    # -------- Linear Model --------
    with mlflow.start_run(run_name="Linear_Regression"):
        linear_pipeline = get_linear_pipeline(preprocessor)
        linear_pipeline.fit(X_train, y_train)

        # Save + log model
        joblib.dump(linear_pipeline, "models/linear_model.pkl")
        mlflow.sklearn.log_model(linear_pipeline, "linear_model", input_example=X_train.head(2))

        # Log metrics
        print("\nLinear Regression Performance:")
        mae, mse, rmse, r2 = evaluate_model(linear_pipeline, X_test, y_test)
        mlflow.log_metrics({"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})

        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)

    # -------- Polynomial Model --------
    with mlflow.start_run(run_name="Polynomial_Regression"):
        poly_pipeline = get_polynomial_pipeline(preprocessor, degree=2)
        poly_pipeline.fit(X_train, y_train)

        # Save + log model
        joblib.dump(poly_pipeline, "models/poly_model.pkl")
        mlflow.sklearn.log_model(poly_pipeline, "polynomial_model", input_example=X_train.head(2))

        # Log metrics
        print("\nPolynomial Regression Performance:")
        mae, mse, rmse, r2 = evaluate_model(poly_pipeline, X_test, y_test)
        mlflow.log_metrics({"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})

        # Log parameters
        mlflow.log_param("model_type", "PolynomialRegression")
        mlflow.log_param("degree", 2)
        mlflow.log_param("test_size", 0.2)

if __name__ == "__main__":
    main()
