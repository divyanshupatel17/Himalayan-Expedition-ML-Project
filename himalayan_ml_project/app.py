import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join('.', 'src'))

# Import custom modules
from demo_model_loader import load_all_models, predict_with_all_models, get_model_performance_summary, get_feature_importance_info

# Himalayan Expedition Success Prediction App
st.set_page_config(
    page_title="🏔️ Himalayan Expedition Success Predictor",
    page_icon="🏔️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f3a5f;
        text-align: center;
    }
    .stAlert {
        background-color: #e1f5fe;
        border: 1px solid #4fc3f7;
        border-radius: 5px;
        padding: 10px;
    }
    .model-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-high {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.2em;
    }
    .prediction-medium {
        color: #f57f17;
        font-weight: bold;
        font-size: 1.2em;
    }
    .prediction-low {
        color: #c62828;
        font-weight: bold;
        font-size: 1.2em;
    }
    .feature-info {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        color: #333333; /* Dark text for better visibility */
    }
    /* Added styles for better text visibility */
    .expedition-details {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #333333;
    }
    .model-comparison {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #333333;
    }
    /* Improved table styling */
    table {
        background-color: #ffffff;
        color: #333333;
    }
    th {
        background-color: #1f3a5f;
        color: white;
        padding: 10px;
    }
    td {
        padding: 8px;
        border-bottom: 1px solid #dddddd;
        color: #333333;
    }
    /* Improved sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    /* Fix for light text on light background */
    .stMarkdown {
        color: #333333;
    }
    .stText {
        color: #333333;
    }
    /* Ensure all text is visible */
    * {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏔️ Himalayan Expedition Success Predictor")
st.markdown("Predict the success of Himalayan climbing expeditions using machine learning")

# Load models once at startup
@st.cache_resource
def load_models():
    """Load all trained models at startup"""
    with st.spinner("Loading machine learning models..."):
        models = load_all_models()
    return models

try:
    models = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Please ensure you've trained all models using the notebooks in notebooks/models/")
    st.stop()

# Sidebar for input parameters
st.sidebar.header("🏔️ Expedition Parameters")

# Input fields for expedition details
age = st.sidebar.slider("Climber Age", 18, 70, 35)
sex = st.sidebar.selectbox("Climber Sex", ["M", "F"])
season = st.sidebar.selectbox("Season", ["Spring", "Autumn", "Winter", "Summer"])
team_size = st.sidebar.slider("Team Size", 1, 20, 5)
hired_staff = st.sidebar.slider("Hired Staff", 0, 15, 3)
peak_height = st.sidebar.slider("Peak Height (meters)", 6000, 8849, 8000)
oxygen_used = st.sidebar.checkbox("Oxygen Used", value=True)
total_members = st.sidebar.slider("Total Members", 1, 20, 5)

# Add a predict button
predict_button = st.sidebar.button("🔮 Predict Success", type="primary")

# Main content area
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Expedition Details")
    st.markdown("<div class='expedition-details'>", unsafe_allow_html=True)
    st.write(f"**Climber Age:** {age} years")
    st.write(f"**Climber Sex:** {sex}")
    st.write(f"**Season:** {season}")
    st.write(f"**Team Size:** {team_size} climbers")
    st.write(f"**Hired Staff:** {hired_staff} support staff")
    st.write(f"**Peak Height:** {peak_height:,} meters")
    st.write(f"**Oxygen Used:** {'Yes' if oxygen_used else 'No'}")
    st.write(f"**Total Members:** {total_members}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature importance information
    st.subheader("🔍 Key Success Factors")
    feature_info = get_feature_importance_info()
    for info in feature_info:
        st.markdown(f"<div class='feature-info'>{info}</div>", unsafe_allow_html=True)

with col2:
    if predict_button or st.session_state.get('predictions', None) is not None:
        if predict_button:
            # Make predictions using all models
            with st.spinner("Making predictions with all models..."):
                try:
                    predictions = predict_with_all_models(
                        models, age, sex, season, team_size, hired_staff,
                        peak_height, oxygen_used, total_members
                    )
                    st.session_state.predictions = predictions
                except Exception as e:
                    st.error(f"Error making predictions: {e}")
                    st.stop()
        else:
            predictions = st.session_state.predictions
        
        st.subheader("🤖 Model Predictions")
        
        # Sort predictions by probability
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Display predictions in cards
        for model_name, probability in sorted_predictions:
            with st.container():
                st.markdown(f"""
                <div class="model-card">
                    <h4>{model_name.replace('_', ' ').title()}</h4>
                    <p>
                """, unsafe_allow_html=True)
                
                # Style based on probability
                if probability > 0.8:
                    st.markdown(f"<span class='prediction-high'>{probability:.2%}</span>", unsafe_allow_html=True)
                elif probability > 0.6:
                    st.markdown(f"<span class='prediction-medium'>{probability:.2%}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='prediction-low'>{probability:.2%}</span>", unsafe_allow_html=True)
                
                st.markdown("</p></div>", unsafe_allow_html=True)
        
        # Overall recommendation
        st.subheader("🏆 Overall Assessment")
        best_model, best_prob = sorted_predictions[0]
        worst_model, worst_prob = sorted_predictions[-1]
        
        if best_prob > 0.8:
            st.success(f"🎉 Excellent chance of success! {best_model.title()} predicts {best_prob:.1%} success probability.")
        elif best_prob > 0.6:
            st.warning(f"⚠️ Moderate chance of success. {best_model.title()} predicts {best_prob:.1%} success probability.")
        else:
            st.error(f"⚠️ Low chance of success. {best_model.title()} predicts {best_prob:.1%} success probability.")
        
        # Model comparison
        st.subheader("📊 Model Comparison")
        st.markdown("<div class='model-comparison'>", unsafe_allow_html=True)
        
        # Create a DataFrame for comparison
        comparison_data = []
        for model_name, probability in sorted_predictions:
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Success Probability': f"{probability:.2%}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Performance summary
        st.subheader("📈 Model Performance Summary")
        performance_summary = get_model_performance_summary()
        
        performance_data = []
        for model_name, metrics in performance_summary.items():
            performance_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.1f}%",
                'Speed': metrics['speed'],
                'Best For': metrics['best_for']
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.table(performance_df)
        
    else:
        st.info("👈 Enter expedition details in the sidebar and click 'Predict Success' to see model predictions.")

# Information about the project
st.markdown("---")
st.subheader("About This Project")
st.markdown("""
This application demonstrates the prediction of Himalayan expedition success using 6 different machine learning algorithms:

1. **XGBoost** - Gradient boosting algorithm with high accuracy
2. **Random Forest** - Ensemble method with consistent performance
3. **LightGBM** - Fast gradient boosting for large datasets
4. **CatBoost** - Categorical boosting with excellent handling of categorical features
5. **Support Vector Machine** - Statistical learning method
6. **Neural Network** - Deep learning approach for complex patterns

**Key Features:**
- Real-time predictions using trained models
- Comparison of all 6 machine learning algorithms
- Performance metrics for each model
- Feature importance analysis
- User-friendly interface for expedition planning

**To use this application:**
1. Adjust the expedition parameters in the sidebar
2. Click the "Predict Success" button
3. View predictions from all models
4. Compare model performance and choose the most reliable prediction
""")

# Footer
st.markdown("---")
st.caption("Himalayan Expedition Success Prediction - Academic Project by Divyanshu Patel (23BAI1214)")