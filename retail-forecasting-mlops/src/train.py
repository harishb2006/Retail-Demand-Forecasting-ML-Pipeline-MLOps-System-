import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model():
    """
    Train the machine learning model and log with MLflow.
    """
    mlflow.set_experiment("retail_forecasting")
    
    with mlflow.start_run():
        # Dummy data for illustration
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y_train = pd.Series([10, 20, 30])
        
        # Train model
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("n_estimators", 10)
        
        # Log and save model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally in models dir
        os.makedirs("../models", exist_ok=True)
        joblib.dump(model, "../models/model.pkl")

if __name__ == "__main__":
    train_model()
