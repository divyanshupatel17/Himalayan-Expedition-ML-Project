# Saved Models Directory

## Overview
This directory contains trained machine learning models for Himalayan expedition success prediction. Each model consists of **3 essential files** that work together.

## File Structure
```
saved_models/
├── xgboost_model.pkl          # Trained XGBoost algorithm
├── xgboost_encoders.pkl       # Categorical variable encoders
├── xgboost_scaler.pkl         # Feature scaling parameters
├── random_forest_model.pkl    # Trained Random Forest algorithm
├── random_forest_encoders.pkl # Categorical variable encoders
├── random_forest_scaler.pkl   # Feature scaling parameters
└── ... (similar pattern for all 6 models)
```

## Why 3 Files Per Model?

### 1. **Model File** (`*_model.pkl`)
- **Contains:** Trained machine learning algorithm
- **Purpose:** Makes predictions on processed data
- **Example:** XGBoost with 100 trees, learned weights and parameters

### 2. **Encoders File** (`*_encoders.pkl`)
- **Contains:** Label encoders for categorical variables
- **Purpose:** Converts text to numbers consistently
- **Example:** 
  - `sex`: F=0, M=1, X=2
  - `season`: Autumn=0, Spring=1, Summer=2, Winter=3

### 3. **Scaler File** (`*_scaler.pkl`)
- **Contains:** StandardScaler with mean and standard deviation
- **Purpose:** Normalizes numerical features
- **Example:** Age scaled using training mean=42.5, std=12.3

## How They Work Together

```python
# Loading for prediction
model = joblib.load('xgboost_model.pkl')
encoders = joblib.load('xgboost_encoders.pkl')
scaler = joblib.load('xgboost_scaler.pkl')

# Processing new data
raw_input = {'age': 35, 'sex': 'M', 'season': 'Spring', ...}
encoded_data = apply_encoders(raw_input, encoders)  # M→1, Spring→1
scaled_data = scaler.transform(encoded_data)        # Normalize numbers
prediction = model.predict(scaled_data)             # Get result
```

## Available Models

| Model | Accuracy | Files |
|-------|----------|-------|
| Random Forest | 79.95% | `random_forest_*.pkl` |
| XGBoost | 79.76% | `xgboost_*.pkl` |
| Neural Network | 79.46% | `neural_network_*.pkl` |
| LightGBM | 79.28% | `lightgbm_*.pkl` |
| CatBoost | 77.45% | `catboost_*.pkl` |
| SVM | 72.54% | `svm_*.pkl` |

## Features Used (All Models)
All models use the same 7 features:
1. **sex** - Climber gender (encoded)
2. **season** - Expedition season (encoded)
3. **heightm** - Peak height in meters
4. **o2used** - Oxygen usage (0/1)
5. **totmembers** - Total team members
6. **age** - Climber age (derived from birth year)
7. **hired_staff** - Number of hired support staff

## Important Notes
- **All 3 files are required** for each model to work properly
- **Never delete or modify** these files individually
- **Consistent preprocessing** is essential for accurate predictions
- Models were trained on 89,089 expedition records with 65.28% success rate

## Usage
These models are automatically loaded by:
- `demo_model_loader.py` - For programmatic access
- `app.py` - For Streamlit dashboard
- Individual notebooks - For training and testing

---
*Generated for Himalayan Expedition Success Prediction Project*