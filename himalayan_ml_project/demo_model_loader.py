"""
Demo Model Loader for Himalayan Expedition Success Prediction
This script demonstrates how to load and use all trained models in the Streamlit dashboard.
"""
import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join('.', 'src'))

# Import custom modules
from src.utils import prepare_features
# We'll modify the load_model function to look in the correct directory
import joblib

def load_model(model_name):
    """
    Load trained model and preprocessors from the correct directory.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        tuple: (model, encoders, scaler)
    """
    # Try multiple possible locations for saved models
    possible_paths = [
        os.path.join('.', 'notebooks', 'saved_models'),
        os.path.join('.', 'saved_models'),
        os.path.join('..', 'saved_models')
    ]
    
    model = None
    encoders = None
    scaler = None
    
    for path in possible_paths:
        try:
            model_path = os.path.join(path, f'{model_name}_model.pkl')
            encoders_path = os.path.join(path, f'{model_name}_encoders.pkl')
            scaler_path = os.path.join(path, f'{model_name}_scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                encoders = joblib.load(encoders_path)
                scaler = joblib.load(scaler_path)
                print(f"Found models in: {path}")
                break
        except Exception as e:
            continue
    
    return model, encoders, scaler

def load_all_models():
    """
    Load all trained models.
    
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    model_names = ['xgboost', 'random_forest', 'lightgbm', 'catboost', 'svm', 'neural_network']
    
    for model_name in model_names:
        try:
            model, encoders, scaler = load_model(model_name)
            if model is not None:
                models[model_name] = {
                    'model': model,
                    'encoders': encoders,
                    'scaler': scaler
                }
                print(f"✓ Loaded {model_name} model successfully")
            else:
                print(f"✗ Failed to load {model_name} model (file not found)")
        except Exception as e:
            print(f"✗ Error loading {model_name} model: {e}")
    
    return models

def prepare_features_for_model(model_name, age, sex, season, team_size, hired_staff, peak_height, oxygen_used, total_members):
    """
    Prepare features exactly as expected by each specific model.
    All models now use the same 7 features based on the actual training data.
    
    Args:
        model_name (str): Name of the model
        age (int): Climber age
        sex (str): Climber sex ('M' or 'F')
        season (str): Season of expedition
        team_size (int): Size of climbing team
        hired_staff (int): Number of hired staff
        peak_height (int): Height of peak in meters
        oxygen_used (bool): Whether oxygen was used
        total_members (int): Total number of members
        
    Returns:
        pd.DataFrame: Prepared features with proper column names for the model
    """
    # All models now use the same 7 features: ['sex', 'season', 'heightm', 'o2used', 'totmembers', 'age', 'hired_staff']
    
    # Encode sex (0=F, 1=M, 2=X - but we only have M/F in input)
    sex_encoded = 1 if sex == 'M' else 0
    
    # Encode season (0=Autumn, 1=Spring, 2=Summer, 3=Winter)
    season_mapping = {'Autumn': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
    season_encoded = season_mapping.get(season, 1)  # Default to Spring
    
    # Convert oxygen_used boolean to int
    o2used_encoded = 1 if oxygen_used else 0
    
    # Create feature DataFrame with proper column names
    feature_data = {
        'sex': [sex_encoded],
        'season': [season_encoded],
        'heightm': [peak_height],
        'o2used': [o2used_encoded],
        'totmembers': [total_members],
        'age': [age],
        'hired_staff': [hired_staff]
    }
    
    return pd.DataFrame(feature_data)

def predict_with_all_models(models, age, sex, season, team_size, hired_staff, peak_height, oxygen_used, total_members):
    """
    Make predictions using all loaded models.
    
    Args:
        models (dict): Dictionary of loaded models
        age (int): Climber age
        sex (str): Climber sex ('M' or 'F')
        season (str): Season of expedition
        team_size (int): Size of climbing team
        hired_staff (int): Number of hired staff
        peak_height (int): Height of peak in meters
        oxygen_used (bool): Whether oxygen was used
        total_members (int): Total number of members
        
    Returns:
        dict: Predictions from all models
    """
    predictions = {}
    
    for model_name, model_data in models.items():
        try:
            # Prepare features exactly as expected by this model
            X = prepare_features_for_model(
                model_name, age, sex, season, team_size, hired_staff, 
                peak_height, oxygen_used, total_members
            )
            
            # Scale features using saved scaler
            X_scaled = model_data['scaler'].transform(X)
            
            # Make prediction
            if model_name == 'svm':
                # SVM uses decision function for probability-like output
                prediction = model_data['model'].decision_function(X_scaled)[0]
                # Convert to probability-like score (0-1 range)
                prediction = 1 / (1 + np.exp(-prediction))
            else:
                # Other models have predict_proba
                try:
                    prediction = model_data['model'].predict_proba(X_scaled)[0][1]
                except:
                    # Fallback to predict method
                    prediction = model_data['model'].predict(X_scaled)[0]
                    # Convert to probability-like score
                    prediction = 0.5 + (prediction * 0.5) if prediction == 1 else 0.5 - (prediction * 0.5)
            
            predictions[model_name] = prediction
        except Exception as e:
            print(f"Error making prediction with {model_name}: {e}")
            predictions[model_name] = 0.0
    
    return predictions

def get_ensemble_prediction(predictions):
    """
    Generate ensemble predictions using different methods.
    
    Args:
        predictions (dict): Dictionary of model predictions
        
    Returns:
        dict: Different ensemble predictions
    """
    if not predictions:
        return {}
    
    values = list(predictions.values())
    
    ensemble_predictions = {
        'simple_average': np.mean(values),
        'weighted_average': np.average(values, weights=[0.2, 0.25, 0.15, 0.15, 0.1, 0.15]),  # Weight based on typical model performance
        'median': np.median(values),
        'conservative': np.percentile(values, 25),  # More conservative estimate
        'optimistic': np.percentile(values, 75),   # More optimistic estimate
    }
    
    return ensemble_predictions

def get_confidence_level(predictions):
    """
    Determine confidence level based on prediction variance.
    
    Args:
        predictions (dict): Dictionary of model predictions
        
    Returns:
        str: Confidence level (High, Medium, Low)
    """
    if not predictions:
        return "Unknown"
    
    values = list(predictions.values())
    std_dev = np.std(values)
    
    if std_dev < 0.1:
        return "High"
    elif std_dev < 0.2:
        return "Medium"
    else:
        return "Low"

def main():
    """
    Main function to demonstrate model loading and prediction.
    """
    print("HIMALAYAN EXPEDITION SUCCESS PREDICTION - MODEL DEMONSTRATION")
    print("=" * 60)
    
    # Load all models
    print("Loading trained models...")
    models = load_all_models()
    
    if not models:
        print("\nNo trained models found. Please follow these steps:")
        print("1. Run each model notebook in notebooks/models/ to train and save models")
        print("2. The notebooks will automatically save models to notebooks/saved_models/ directory")
        print("3. After running all notebooks, run this script again")
        print("\nExpected model files in notebooks/saved_models/ directory:")
        model_names = ['xgboost', 'random_forest', 'lightgbm', 'catboost', 'svm', 'neural_network']
        for name in model_names:
            print(f"  - {name}_model.pkl")
            print(f"  - {name}_encoders.pkl")
            print(f"  - {name}_scaler.pkl")
        return
    
    print(f"\nSuccessfully loaded {len(models)} models:")
    for model_name in models.keys():
        print(f"  - {model_name}")
    
    # Example prediction with sample data
    print("\nMaking sample predictions...")
    print("-" * 30)
    
    # Sample user input (typical successful expedition)
    age = 35
    sex = 'M'
    season = 'Spring'
    team_size = 5
    hired_staff = 8
    peak_height = 8000
    oxygen_used = True
    total_members = 10
    
    print("Sample Expedition Details:")
    print("  Age: 35, Sex: M, Season: Spring")
    print("  Team Size: 5, Hired Staff: 8")
    print("  Peak Height: 8000m, Oxygen Used: Yes, Total Members: 10")
    
    predictions = predict_with_all_models(
        models, age, sex, season, team_size, hired_staff, 
        peak_height, oxygen_used, total_members
    )
    
    print("\nSuccess Probability Predictions:")
    
    # Sort predictions by probability
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    for model_name, probability in sorted_predictions:
        print(f"  {model_name:15s}: {probability:.2%}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("For your Streamlit dashboard:")
    print("1. Load models once at startup using load_all_models()")
    print("2. For each user prediction, use predict_with_all_models()")
    print("3. Display results sorted by confidence")
    print("4. Highlight top-performing models (XGBoost, Random Forest)")

# Additional utility functions for the Streamlit dashboard
def get_model_performance_summary():
    """
    Get a summary of model performance for display in the dashboard.
    
    Returns:
        dict: Performance summary
    """
    # These are the results from our final verification
    return {
        'XGBoost': {'accuracy': 85.13, 'speed': 'Fast', 'best_for': 'High accuracy'},
        'Random Forest': {'accuracy': 84.87, 'speed': 'Fast', 'best_for': 'Consistent performance'},
        'CatBoost': {'accuracy': 82.42, 'speed': 'Medium', 'best_for': 'Categorical features'},
        'LightGBM': {'accuracy': 81.80, 'speed': 'Fast', 'best_for': 'Large datasets'},
        'Neural Network': {'accuracy': 79.46, 'speed': 'Medium', 'best_for': 'Complex patterns'},
        'SVM': {'accuracy': 72.71, 'speed': 'Slow', 'best_for': 'Small datasets'}
    }

def get_feature_importance_info():
    """
    Get information about important features for display in the dashboard.
    
    Returns:
        list: Feature importance information
    """
    return [
        "Key factors affecting expedition success:",
        "1. Oxygen usage - Most important factor",
        "2. Peak height - Critical difficulty indicator",
        "3. Team composition - Size and experience matter",
        "4. Season - Weather conditions affect success",
        "5. Hired staff - Support personnel improve chances"
    ]

if __name__ == "__main__":
    main()