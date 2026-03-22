import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os

from src.ingestion import load_data
from src.preprocessing import clean_data
from src.features import generate_features

def train_model():
    """
    Train the machine learning model and log with MLflow.
    """
    mlflow.set_experiment("retail_forecasting")
    
    with mlflow.start_run():
        # Pipeline execution
        data_path = "data/store-sales-time-series-forecasting/train.csv"
        print(f"Loading data from {data_path}...")
        df = load_data(data_path)
        
        print("Preprocessing data...")
        df = clean_data(df)
        
        print("Generating features...")
        df = generate_features(df)
        
        # Drop rows with NaN resulting from shift and rolling
        df = df.dropna()
        
        # Target and features setup
        features = ['lag_1', 'rolling_mean'] 
        if 'store_nbr' in df.columns:
            features.append('store_nbr')
            
        X = df[features]
        y = df['sales']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model on a subset to ensure this runs fast
        X_train_sub = X_train.head(10000)
        y_train_sub = y_train.head(10000)
        X_test_sub = X_test.head(2000)
        y_test_sub = y_test.head(2000)
        
        # --- Baseline Model ---
        print("Training Linear Regression (Baseline)...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_sub, y_train_sub)
        
        lr_preds = lr_model.predict(X_test_sub)
        lr_rmse = np.sqrt(mean_squared_error(y_test_sub, lr_preds))
        lr_mae = mean_absolute_error(y_test_sub, lr_preds)
        print(f"Linear Regression - RMSE: {lr_rmse:.2f}, MAE: {lr_mae:.2f}")

        # --- Better Model ---
        print("Training XGBoost...")
        xgb_model = XGBRegressor(n_estimators=50, max_depth=5, random_state=42)
        xgb_model.fit(X_train_sub, y_train_sub)
        
        xgb_preds = xgb_model.predict(X_test_sub)
        xgb_rmse = np.sqrt(mean_squared_error(y_test_sub, xgb_preds))
        xgb_mae = mean_absolute_error(y_test_sub, xgb_preds)
        print(f"XGBoost - RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}")
        
        # Log parameters and metrics for the better model
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", xgb_rmse)
        mlflow.log_metric("mae", xgb_mae)
        
        # Log and save models
        mlflow.sklearn.log_model(lr_model, "baseline_model")
        mlflow.xgboost.log_model(xgb_model, "model")
        
        # Save XGBoost model locally in models dir
        os.makedirs("models", exist_ok=True)
        joblib.dump(xgb_model, "models/model.pkl")
        print("Training completed and models saved!")

if __name__ == "__main__":
    train_model()
