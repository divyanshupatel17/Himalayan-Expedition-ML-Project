# Himalayan Expedition Success Prediction - Minimal Setup

## Project Overview
This is a minimal setup for the Himalayan Expedition Success Prediction project with all 6 machine learning models organized in separate notebooks for educational demonstration.

## Minimal Directory Structure

```
himalayan_ml_minimal/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app.py                       # Streamlit dashboard
├── demo_model_loader.py         # Model loading utilities
├── final_model_verification.py  # Model verification script
├── MODEL_IMPROVEMENTS_SUMMARY.txt # Model improvements documentation
├── data/                        # Data storage
│   ├── expeditions.csv         # Raw expedition data
│   ├── members.csv             # Raw member data
│   └── peaks.csv               # Raw peak data
├── notebooks/                  # Jupyter notebooks (one per model)
│   ├── models/                 # Individual model implementations
│   │   ├── 01_xgboost.ipynb    # XGBoost model
│   │   ├── 02_random_forest.ipynb  # Random Forest model
│   │   ├── 03_lightgbm.ipynb   # LightGBM model
│   │   ├── 04_catboost.ipynb   # CatBoost model
│   │   ├── 05_svm.ipynb        # Support Vector Machine
│   │   └── 06_neural_network.ipynb  # Neural Network
│   └── saved_models/           # Saved trained models and preprocessors
│       ├── xgboost_model.pkl
│       ├── xgboost_encoders.pkl
│       ├── xgboost_scaler.pkl
│       ├── random_forest_model.pkl
│       ├── random_forest_encoders.pkl
│       ├── random_forest_scaler.pkl
│       ├── lightgbm_model.pkl
│       ├── lightgbm_encoders.pkl
│       ├── lightgbm_scaler.pkl
│       ├── catboost_model.pkl
│       ├── catboost_encoders.pkl
│       ├── catboost_scaler.pkl
│       ├── svm_model.pkl
│       ├── svm_encoders.pkl
│       ├── svm_scaler.pkl
│       ├── neural_network_model.pkl
│       ├── neural_network_encoders.pkl
│       └── neural_network_scaler.pkl
├── src/                        # Source code
│   ├── data_loader.py          # Data loading functions
│   └── utils.py                # Utility functions
└── __pycache__/                # Python cache files
```

## Features of This Minimal Setup

1. **Offline Operation**: No cloud services or internet dependencies
2. **Simple Structure**: Easy to understand and navigate
3. **Separate Notebooks**: Each model in its own notebook for clarity
4. **Minimal Dependencies**: Only essential libraries
5. **Educational Focus**: Clear code for classroom demonstration
6. **All 6 Models**: Complete implementation of all required algorithms

## Requirements

- Python 3.7+
- Jupyter Notebook
- Basic Python libraries (pandas, numpy, scikit-learn)
- Model-specific libraries (xgboost, lightgbm, catboost)
- Streamlit for dashboard

## Setup Instructions

1. Place the Himalayan expedition dataset files in the `data/` directory
2. Install requirements: `pip install -r requirements.txt`
3. Run notebooks individually to train models:
   - Open each notebook in `notebooks/models/` and execute all cells
   - Models will be automatically saved to `notebooks/saved_models/`
4. Start dashboard: `streamlit run app.py`

## For Classroom Demonstration

This structure allows you to:
1. Show each model separately in its notebook
2. Compare model performance
3. Demonstrate predictions in the Streamlit app
4. Explain differences between algorithms

## Files Overview

- **app.py**: Main Streamlit application for interactive predictions
- **demo_model_loader.py**: Utility functions for loading and using trained models
- **final_model_verification.py**: Script to verify all models are working correctly
- **MODEL_IMPROVEMENTS_SUMMARY.txt**: Documentation of model improvements and findings
- **notebooks/models/*.ipynb**: Individual notebooks for each machine learning algorithm
- **src/data_loader.py**: Functions for loading and preprocessing the expedition data
- **src/utils.py**: Utility functions for feature preparation, model saving/loading, and evaluation