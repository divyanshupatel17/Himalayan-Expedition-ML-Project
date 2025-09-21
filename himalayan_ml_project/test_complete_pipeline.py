"""
Complete pipeline test for Himalayan ML Project
Tests data loading, feature preparation, and model functionality
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join('.', 'src'))

def test_data_loading():
    """Test data loading functionality"""
    print("="*50)
    print("TESTING DATA LOADING")
    print("="*50)
    
    try:
        from data_loader import load_data, create_master_dataset
        
        # Load data
        expeditions, members, peaks = load_data()
        
        if expeditions is not None:
            print(f"✓ Expeditions loaded: {expeditions.shape}")
            print(f"✓ Members loaded: {members.shape}")
            print(f"✓ Peaks loaded: {peaks.shape}")
            
            # Create master dataset
            master_df = create_master_dataset(expeditions, members, peaks)
            if master_df is not None:
                print(f"✓ Master dataset created: {master_df.shape}")
                print(f"✓ Columns: {list(master_df.columns)}")
                return master_df
            else:
                print("✗ Failed to create master dataset")
                return None
        else:
            print("✗ Failed to load data")
            return None
            
    except Exception as e:
        print(f"✗ Error in data loading: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_feature_preparation(df):
    """Test feature preparation"""
    print("\n" + "="*50)
    print("TESTING FEATURE PREPARATION")
    print("="*50)
    
    try:
        from utils import prepare_features
        
        X, y, encoders = prepare_features(df)
        
        print(f"✓ Features prepared successfully")
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Target vector shape: {y.shape}")
        print(f"✓ Encoders created: {list(encoders.keys())}")
        
        return X, y, encoders
        
    except Exception as e:
        print(f"✗ Error in feature preparation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_model_loading():
    """Test model loading functionality"""
    print("\n" + "="*50)
    print("TESTING MODEL LOADING")
    print("="*50)
    
    try:
        from demo_model_loader import load_all_models
        
        models = load_all_models()
        
        if models:
            print(f"✓ Successfully loaded {len(models)} models:")
            for model_name in models.keys():
                print(f"  - {model_name}")
            return models
        else:
            print("✗ No models loaded")
            return None
            
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_predictions(models):
    """Test prediction functionality"""
    print("\n" + "="*50)
    print("TESTING PREDICTIONS")
    print("="*50)
    
    try:
        from demo_model_loader import predict_with_all_models
        
        # Sample expedition parameters
        predictions = predict_with_all_models(
            models=models,
            age=35,
            sex='M',
            season='Spring',
            team_size=5,
            hired_staff=8,
            peak_height=8000,
            oxygen_used=True,
            total_members=10
        )
        
        print("✓ Predictions generated successfully:")
        for model_name, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name:15s}: {prob:.2%}")
            
        return predictions
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_streamlit_app():
    """Test if Streamlit app can be imported"""
    print("\n" + "="*50)
    print("TESTING STREAMLIT APP")
    print("="*50)
    
    try:
        # Check if app.py exists and can be parsed
        app_path = 'app.py'
        if os.path.exists(app_path):
            with open(app_path, 'r') as f:
                content = f.read()
            
            # Basic syntax check
            compile(content, app_path, 'exec')
            print("✓ Streamlit app syntax is valid")
            
            # Check for required imports
            required_imports = ['streamlit', 'pandas', 'numpy']
            for imp in required_imports:
                if f'import {imp}' in content:
                    print(f"✓ {imp} import found")
                else:
                    print(f"⚠ {imp} import not found")
            
            return True
        else:
            print("✗ app.py not found")
            return False
            
    except Exception as e:
        print(f"✗ Error checking Streamlit app: {e}")
        return False

def main():
    """Run all tests"""
    print("HIMALAYAN ML PROJECT - COMPLETE PIPELINE TEST")
    print("="*60)
    
    # Test 1: Data Loading
    master_df = test_data_loading()
    if master_df is None:
        print("\n❌ CRITICAL: Data loading failed. Cannot proceed with other tests.")
        return
    
    # Test 2: Feature Preparation
    X, y, encoders = test_feature_preparation(master_df)
    if X is None:
        print("\n❌ CRITICAL: Feature preparation failed.")
        return
    
    # Test 3: Model Loading
    models = test_model_loading()
    if models is None:
        print("\n⚠ WARNING: Model loading failed. Models may not be trained yet.")
    else:
        # Test 4: Predictions
        predictions = test_predictions(models)
        if predictions is None:
            print("\n❌ CRITICAL: Prediction functionality failed.")
    
    # Test 5: Streamlit App
    app_ok = test_streamlit_app()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if master_df is not None:
        print("✓ Data loading: PASSED")
    else:
        print("✗ Data loading: FAILED")
    
    if X is not None:
        print("✓ Feature preparation: PASSED")
    else:
        print("✗ Feature preparation: FAILED")
    
    if models is not None:
        print("✓ Model loading: PASSED")
    else:
        print("✗ Model loading: FAILED")
    
    if models is not None and 'predictions' in locals() and predictions is not None:
        print("✓ Predictions: PASSED")
    else:
        print("✗ Predictions: FAILED")
    
    if app_ok:
        print("✓ Streamlit app: PASSED")
    else:
        print("✗ Streamlit app: FAILED")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if master_df is None:
        print("1. Check that dataset files exist in data/ directory")
        print("2. Verify CSV files are not corrupted")
    
    if models is None:
        print("3. Run all notebook files in notebooks/models/ to train models")
        print("4. Ensure models are saved to notebooks/saved_models/")
    
    print("5. To run the Streamlit app: streamlit run app.py")
    print("6. Make sure all requirements are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()