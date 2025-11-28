# Prediction module
import pandas as pd
import numpy as np
import joblib

def load_model():
    """Load the trained model"""
    model = joblib.load('../models/final_best_model.pkl')
    return model

def predict_heart_disease(model, patient_data):
    """
    Make prediction for a patient
    patient_data: dictionary with patient features
    """
    # Convert to dataframe
    df = pd.DataFrame([patient_data])
    
    # Make prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    return prediction[0], probability[0]

# Test function
if __name__ == "__main__":
    print("Predict module loaded successfully")