import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
            
        X_train = df[features]
        y_train = df['sales']
        
        # Train model on a subset to ensure this runs fast
        print("Training model...")
        X_train_sub = X_train.head(10000)
        y_train_sub = y_train.head(10000)
        
        model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train_sub, y_train_sub)
        
        # Log parameters
        mlflow.log_param("n_estimators", 10)
        mlflow.log_param("max_depth", 5)
        
        # Log and save model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally in models dir
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        print("Training completed and model saved!")

if __name__ == "__main__":
    train_model()
