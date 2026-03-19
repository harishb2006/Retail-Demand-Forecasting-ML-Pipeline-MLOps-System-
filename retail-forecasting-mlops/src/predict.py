import joblib
import pandas as pd
import os

def load_model():
    """
    Load the trained model from the models directory.
    """
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def make_prediction(input_data: dict):
    """
    Make predictions using the loaded model.
    """
    model = load_model()
    if model is None:
        raise FileNotFoundError("Model not found. Please train the model first.")
        
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction.tolist()
