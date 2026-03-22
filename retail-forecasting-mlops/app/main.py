from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI(title="Retail Forecasting MLOps API", version="1.0")

# Attempt to load model
model_path = "models/model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    # Handle the case where the model hasn't been trained yet
    model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to Retail Forecasting MLOps API"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not found. Please train the model first."}
        
    # convert to dataframe
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
