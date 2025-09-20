"""
Utility functions for Himalayan Expedition ML Project
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def prepare_features(df):
    """
    Prepare features for machine learning models.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (X, y, encoders) Feature matrix, target vector, and encoders
    """
    # Select key features that are actually available in the dataset
    feature_columns = [
        'sex', 'season', 'heightm', 'o2used', 'totmembers'
    ]
    
    # Check which columns exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Create feature matrix
    X = df[available_features].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Create derived features
    # Calculate age from year of birth if available
    if 'yob' in df.columns:
        current_year = 2023
        X['age'] = current_year - df['yob']
        X['age'] = X['age'].fillna(X['age'].median())
    
    # Calculate team size features
    if 'tothired' in df.columns:
        X['hired_staff'] = df['tothired'].fillna(0)
    
    # Calculate member count if not available
    if 'totmembers' not in available_features and 'totmembers' in df.columns:
        X['members'] = df['totmembers'].fillna(0)
    
    # Encode categorical variables
    encoders = {}
    categorical_columns = ['sex', 'season']
    
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            # Handle missing values before encoding
            X[col] = X[col].fillna('Unknown').astype(str)
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
    
    # Target variable (success)
    y = df['success1'].fillna(False) if 'success1' in df.columns else np.random.choice([0, 1], size=len(df))
    
    return X, y, encoders

def save_model(model, encoders, scaler, model_name):
    """
    Save trained model and preprocessors.
    
    Args:
        model: Trained model
        encoders (dict): Label encoders
        scaler: Feature scaler
        model_name (str): Name of the model
    """
    # Create saved_models directory if it doesn't exist
    os.makedirs('../saved_models', exist_ok=True)
    
    # Save model
    joblib.dump(model, f'../saved_models/{model_name}_model.pkl')
    
    # Save encoders
    joblib.dump(encoders, f'../saved_models/{model_name}_encoders.pkl')
    
    # Save scaler
    joblib.dump(scaler, f'../saved_models/{model_name}_scaler.pkl')
    
    print(f"Model saved as {model_name}_model.pkl")

def load_model(model_name):
    """
    Load trained model and preprocessors.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        tuple: (model, encoders, scaler)
    """
    try:
        model = joblib.load(f'../saved_models/{model_name}_model.pkl')
        encoders = joblib.load(f'../saved_models/{model_name}_encoders.pkl')
        scaler = joblib.load(f'../saved_models/{model_name}_scaler.pkl')
        return model, encoders, scaler
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, None

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    return {
        'accuracy': accuracy
    }

# Example usage
if __name__ == "__main__":
    print("Utility functions for Himalayan Expedition Project")