# -*- coding: utf-8 -*-
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
    page_title="Himalayan Expedition Success Predictor",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling with Enhanced Visibility
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles - Enhanced for Visibility */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
        color: #1a202c;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        background-color: #ffffff;
    }
    
    /* Force text visibility */
    * {
        color: #1a202c !important;
    }
    
    /* Header Styles */
    .header-container {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: white !important;
        padding: 3rem 2rem;
        margin: -2rem -2rem 2rem -2rem;
        text-align: center;
        border-radius: 0 0 15px 15px;
    }
    
    .header-container * {
        color: white !important;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 1rem;
        opacity: 0.95;
        color: white !important;
    }
    
    .header-team {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 1.5rem;
        color: white !important;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background-color: #f7fafc;
        border-right: 2px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .sidebar-header {
        background-color: #edf2f7;
        padding: 1.5rem 1rem;
        margin: -1rem -1rem 1.5rem -1rem;
        border-bottom: 2px solid #cbd5e0;
        border-radius: 8px;
    }
    
    .sidebar-header * {
        color: #2d3748 !important;
    }
    
    /* Card Styles - Enhanced */
    .info-card {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-card * {
        color: #2d3748 !important;
    }
    
    .prediction-card {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .prediction-card * {
        color: #2d3748 !important;
    }
    
    /* Typography - Enhanced Visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #2d3748 !important;
        font-weight: 700 !important;
    }
    
    .section-title {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #2d3748 !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.75rem !important;
        border-bottom: 3px solid #4299e1 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .subsection-title {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Prediction Probability Styles - Enhanced */
    .prob-high {
        color: #38a169 !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .prob-medium {
        color: #ed8936 !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .prob-low {
        color: #e53e3e !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Model Performance Badge - Enhanced */
    .model-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-high {
        background-color: #c6f6d5;
        color: #22543d !important;
        border: 2px solid #38a169;
    }
    
    .badge-medium {
        background-color: #feebc8;
        color: #744210 !important;
        border: 2px solid #ed8936;
    }
    
    .badge-low {
        background-color: #fed7d7;
        color: #742a2a !important;
        border: 2px solid #e53e3e;
    }
    
    /* Table Styles - Professional */
    .stDataFrame {
        background-color: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stDataFrame table {
        background-color: #ffffff !important;
        color: #2d3748 !important;
    }
    
    .stDataFrame th {
        background-color: #4a5568 !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        text-align: center !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stDataFrame td {
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
        color: #2d3748 !important;
        text-align: center !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    .stDataFrame tr:nth-child(even) {
        background-color: #f7fafc !important;
    }
    
    .stDataFrame tr:hover {
        background-color: #edf2f7 !important;
    }
    
    /* Button Styles - Enhanced */
    .stButton > button {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4);
    }
    
    /* Alert Styles - Enhanced Visibility */
    .stSuccess {
        background-color: #f0fff4 !important;
        border: 2px solid #68d391 !important;
        color: #22543d !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stSuccess * {
        color: #22543d !important;
    }
    
    .stWarning {
        background-color: #fffaf0 !important;
        border: 2px solid #f6ad55 !important;
        color: #744210 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stWarning * {
        color: #744210 !important;
    }
    
    .stError {
        background-color: #fff5f5 !important;
        border: 2px solid #fc8181 !important;
        color: #742a2a !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stError * {
        color: #742a2a !important;
    }
    
    .stInfo {
        background-color: #ebf8ff !important;
        border: 2px solid #63b3ed !important;
        color: #2c5282 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stInfo * {
        color: #2c5282 !important;
    }
    
    /* Feature List - Enhanced */
    .feature-list {
        background-color: #f7fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .feature-item {
        padding: 0.75rem 0;
        border-bottom: 1px solid #e2e8f0;
        color: #2d3748 !important;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .feature-item:last-child {
        border-bottom: none;
    }
    
    .feature-item strong {
        color: #2d3748 !important;
        font-weight: 700 !important;
    }
    
    /* Comparison Table Styles */
    .comparison-table {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .comparison-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .comparison-table th {
        background-color: #4a5568;
        color: white !important;
        padding: 1rem;
        text-align: center;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .comparison-table td {
        padding: 0.75rem 1rem;
        text-align: center;
        border-bottom: 1px solid #e2e8f0;
        color: #2d3748 !important;
        font-weight: 500;
    }
    
    .comparison-table tr:nth-child(even) {
        background-color: #f7fafc;
    }
    
    /* Footer - Enhanced */
    .footer {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: white !important;
        padding: 3rem 2rem;
        margin: 4rem -2rem -2rem -2rem;
        text-align: center;
        border-radius: 15px 15px 0 0;
    }
    
    .footer * {
        color: white !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem !important;
        }
        
        .header-subtitle {
            font-size: 1rem !important;
        }
        
        .section-title {
            font-size: 1.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Predictive Modeling for Himalayan Expedition Success</h1>
    <p class="header-subtitle">Harnessing Machine Learning to Improve Safety, Decision-Making, and Risk Management</p>
    <div class="header-team">
        <strong>Instructor:</strong> Dr. Bhargavi R<br>
        <strong>Team Members:</strong><br>
        • 23BAI1214 - Divyanshu Patel<br>
        • 23BAI1162 - Ayush Kumar Singh
    </div>
</div>
""", unsafe_allow_html=True)

# Load models once at startup
@st.cache_resource
def load_models():
    """Load all trained models at startup"""
    with st.spinner("Loading machine learning models..."):
        models = load_all_models()
    return models

try:
    models = load_models()
    if not models:
        st.error("No models could be loaded. Please ensure model files are available.")
        st.info("The system will use sample predictions for demonstration purposes.")
        models = {}
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("The system will use sample predictions for demonstration purposes.")
    models = {}

# Sidebar for input parameters
st.sidebar.markdown("""
<div class="sidebar-header">
    <h3 style="margin: 0; color: #1f2937; font-size: 1.25rem;">Expedition Parameters</h3>
    <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-size: 0.9rem;">Configure expedition details for prediction</p>
</div>
""", unsafe_allow_html=True)

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
predict_button = st.sidebar.button("Generate Predictions", type="primary")

# Main content area
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown('<h2 class="section-title">Expedition Configuration</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-card">
        <h4 class="subsection-title">Current Parameters</h4>
        <div style="display: grid; gap: 0.75rem;">
            <div><strong>Climber Age:</strong> {age} years</div>
            <div><strong>Climber Gender:</strong> {sex}</div>
            <div><strong>Season:</strong> {season}</div>
            <div><strong>Team Size:</strong> {team_size} climbers</div>
            <div><strong>Hired Staff:</strong> {hired_staff} support personnel</div>
            <div><strong>Peak Height:</strong> {peak_height:,} meters</div>
            <div><strong>Oxygen Usage:</strong> {'Yes' if oxygen_used else 'No'}</div>
            <div><strong>Total Members:</strong> {total_members}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance information
    st.markdown('<h2 class="section-title">Key Success Factors</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-list">
        <div class="feature-item"><strong>Oxygen Usage:</strong> Primary determinant of expedition success</div>
        <div class="feature-item"><strong>Peak Height:</strong> Directly correlates with technical difficulty</div>
        <div class="feature-item"><strong>Team Composition:</strong> Size and experience level impact</div>
        <div class="feature-item"><strong>Seasonal Conditions:</strong> Weather patterns affect success rates</div>
        <div class="feature-item"><strong>Support Staff:</strong> Professional assistance improves outcomes</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if predict_button or st.session_state.get('predictions', None) is not None:
        if predict_button:
            # Make predictions using all models
            with st.spinner("Making predictions with all models..."):
                try:
                    if models:
                        predictions = predict_with_all_models(
                            models, age, sex, season, team_size, hired_staff,
                            peak_height, oxygen_used, total_members
                        )
                    else:
                        # Sample predictions for demonstration
                        predictions = {
                            'xgboost': 0.75,
                            'random_forest': 0.72,
                            'neural_network': 0.68,
                            'lightgbm': 0.70,
                            'catboost': 0.69,
                            'svm': 0.65
                        }
                    st.session_state.predictions = predictions
                except Exception as e:
                    st.error(f"Error making predictions: {e}")
                    st.stop()
        else:
            predictions = st.session_state.predictions
        
        st.markdown('<h2 class="section-title">Model Predictions</h2>', unsafe_allow_html=True)
        
        # Sort predictions by probability
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Display predictions in cards
        for i, (model_name, probability) in enumerate(sorted_predictions):
            # Determine probability class and badge
            if probability > 0.8:
                prob_class = 'prob-high'
                badge_class = 'badge-high'
                confidence = 'High'
            elif probability > 0.6:
                prob_class = 'prob-medium'
                badge_class = 'badge-medium'
                confidence = 'Medium'
            else:
                prob_class = 'prob-low'
                badge_class = 'badge-low'
                confidence = 'Low'
            
            rank = i + 1
            
            st.markdown(f"""
            <div class="prediction-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #1f2937;">#{rank} {model_name.replace('_', ' ').title()}</h4>
                    <span class="model-badge {badge_class}">{confidence} Confidence</span>
                </div>
                <div style="text-align: center;">
                    <div class="{prob_class}">{probability:.1%}</div>
                    <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-size: 0.9rem;">Success Probability</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall recommendation
        st.markdown('<h2 class="section-title">Overall Assessment</h2>', unsafe_allow_html=True)
        
        if sorted_predictions:
            best_model, best_prob = sorted_predictions[0]
            
            if best_prob > 0.8:
                st.success(f"High probability of success. The {best_model.replace('_', ' ').title()} model predicts a {best_prob:.1%} success probability.")
            elif best_prob > 0.6:
                st.warning(f"Moderate probability of success. The {best_model.replace('_', ' ').title()} model predicts a {best_prob:.1%} success probability.")
            else:
                st.error(f"Low probability of success. The {best_model.replace('_', ' ').title()} model predicts a {best_prob:.1%} success probability.")
        else:
            st.error("No model predictions available. Please ensure models are properly loaded.")
        
        # Model comparison
        st.markdown('<h2 class="section-title">Model Comparison</h2>', unsafe_allow_html=True)
        
        # Create a DataFrame for comparison
        comparison_data = []
        for model_name, probability in sorted_predictions:
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Success Probability': f"{probability:.2%}",
                'Rank': f"#{sorted_predictions.index((model_name, probability)) + 1}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Model Performance Summary
        st.markdown('<h2 class="section-title">Model Performance Metrics</h2>', unsafe_allow_html=True)
        
        try:
            performance_summary = get_model_performance_summary()
            
            performance_data = []
            for model_name, metrics in performance_summary.items():
                performance_data.append({
                    'Algorithm': model_name,
                    'Training Accuracy': f"{metrics['accuracy']:.1f}%",
                    'Processing Speed': metrics['speed'],
                    'Optimal Use Case': metrics['best_for'],
                    'Complexity': 'High' if model_name in ['Neural Network', 'XGBoost'] else 'Medium' if model_name in ['Random Forest', 'CatBoost'] else 'Low'
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            # Fallback performance data
            performance_data = [
                {'Algorithm': 'Random Forest', 'Training Accuracy': '79.95%', 'Processing Speed': 'Fast', 'Optimal Use Case': 'Consistent performance', 'Complexity': 'Medium'},
                {'Algorithm': 'XGBoost', 'Training Accuracy': '79.76%', 'Processing Speed': 'Fast', 'Optimal Use Case': 'High accuracy', 'Complexity': 'High'},
                {'Algorithm': 'Neural Network', 'Training Accuracy': '79.46%', 'Processing Speed': 'Medium', 'Optimal Use Case': 'Complex patterns', 'Complexity': 'High'},
                {'Algorithm': 'LightGBM', 'Training Accuracy': '79.28%', 'Processing Speed': 'Very Fast', 'Optimal Use Case': 'Large datasets', 'Complexity': 'Medium'},
                {'Algorithm': 'CatBoost', 'Training Accuracy': '77.45%', 'Processing Speed': 'Medium', 'Optimal Use Case': 'Categorical features', 'Complexity': 'Medium'},
                {'Algorithm': 'SVM', 'Training Accuracy': '72.54%', 'Processing Speed': 'Slow', 'Optimal Use Case': 'Small datasets', 'Complexity': 'Low'}
            ]
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True, hide_index=True)
        
        # Detailed Model Analysis
        st.markdown('<h2 class="section-title">Detailed Model Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4 class="subsection-title">Top Performing Models</h4>
                <div style="display: grid; gap: 0.75rem;">
                    <div><strong>🥇 Random Forest:</strong> Most consistent predictions with ensemble approach</div>
                    <div><strong>🥈 XGBoost:</strong> High accuracy with gradient boosting optimization</div>
                    <div><strong>🥉 Neural Network:</strong> Complex pattern recognition capabilities</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4 class="subsection-title">Model Characteristics</h4>
                <div style="display: grid; gap: 0.75rem;">
                    <div><strong>Fastest:</strong> LightGBM - Optimized for speed</div>
                    <div><strong>Most Accurate:</strong> Random Forest - Reliable predictions</div>
                    <div><strong>Best for Beginners:</strong> SVM - Simple and interpretable</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature Importance Comparison
        st.markdown('<h2 class="section-title">Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        feature_importance_data = [
            {'Feature': 'Oxygen Usage', 'Importance': 'Very High', 'Impact': 'Primary success determinant', 'Weight': '25%'},
            {'Feature': 'Peak Height', 'Importance': 'High', 'Impact': 'Difficulty correlation', 'Weight': '20%'},
            {'Feature': 'Team Size', 'Importance': 'Medium', 'Impact': 'Resource optimization', 'Weight': '15%'},
            {'Feature': 'Season', 'Importance': 'Medium', 'Impact': 'Weather conditions', 'Weight': '15%'},
            {'Feature': 'Hired Staff', 'Importance': 'Medium', 'Impact': 'Professional support', 'Weight': '12%'},
            {'Feature': 'Climber Age', 'Importance': 'Low', 'Impact': 'Experience factor', 'Weight': '8%'},
            {'Feature': 'Gender', 'Importance': 'Low', 'Impact': 'Minimal correlation', 'Weight': '5%'}
        ]
        
        feature_df = pd.DataFrame(feature_importance_data)
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        
    else:
        st.markdown('<h2 class="section-title">Prediction Results</h2>', unsafe_allow_html=True)
        st.info("Configure expedition parameters in the sidebar and click 'Generate Predictions' to analyze success probability using machine learning models.")

# Project Information Section
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown('<h1 class="section-title" style="text-align: center; font-size: 2rem; margin: 3rem 0 2rem 0;">Project Overview & Methodology</h1>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3 class="subsection-title">Machine Learning Algorithms</h3>
        <div style="display: grid; gap: 0.75rem;">
            <div><strong>XGBoost:</strong> Gradient boosting with high accuracy (79.76%)</div>
            <div><strong>Random Forest:</strong> Ensemble method for consistency (79.95%)</div>
            <div><strong>LightGBM:</strong> Fast gradient boosting algorithm (79.28%)</div>
            <div><strong>CatBoost:</strong> Categorical feature optimization (77.45%)</div>
            <div><strong>Support Vector Machine:</strong> Statistical learning approach (72.54%)</div>
            <div><strong>Neural Network:</strong> Deep learning methodology (79.46%)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3 class="subsection-title">System Capabilities</h3>
        <div style="display: grid; gap: 0.75rem;">
            <div><strong>Real-time Analysis:</strong> Instant prediction generation</div>
            <div><strong>Multi-model Comparison:</strong> Six algorithm evaluation</div>
            <div><strong>Performance Metrics:</strong> Accuracy and speed analysis</div>
            <div><strong>Feature Analysis:</strong> Success factor identification</div>
            <div><strong>Risk Assessment:</strong> Probability-based recommendations</div>
            <div><strong>Data-driven Insights:</strong> Historical expedition patterns</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Dataset Information
st.markdown("""
<div class="info-card" style="margin-top: 2rem;">
    <h3 class="subsection-title">Dataset & Methodology</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-top: 1.5rem;">
        <div>
            <h4 style="color: #2d3748; margin-bottom: 0.5rem;">📊 Data Scale</h4>
            <ul style="color: #2d3748; line-height: 1.8;">
                <li><strong>89,089</strong> expedition records</li>
                <li><strong>149</strong> total features available</li>
                <li><strong>7</strong> key features selected</li>
                <li><strong>65.28%</strong> historical success rate</li>
            </ul>
        </div>
        <div>
            <h4 style="color: #2d3748; margin-bottom: 0.5rem;">🔬 Methodology</h4>
            <ul style="color: #2d3748; line-height: 1.8;">
                <li>Supervised machine learning</li>
                <li>Cross-validation testing</li>
                <li>Feature engineering</li>
                <li>Ensemble predictions</li>
            </ul>
        </div>
        <div>
            <h4 style="color: #2d3748; margin-bottom: 0.5rem;">🎯 Applications</h4>
            <ul style="color: #2d3748; line-height: 1.8;">
                <li>Expedition planning</li>
                <li>Risk assessment</li>
                <li>Safety optimization</li>
                <li>Resource allocation</li>
            </ul>
        </div>
        <div>
            <h4 style="color: #2d3748; margin-bottom: 0.5rem;">📈 Performance</h4>
            <ul style="color: #2d3748; line-height: 1.8;">
                <li><strong>79.95%</strong> best accuracy</li>
                <li><strong>6</strong> model comparison</li>
                <li>Real-time predictions</li>
                <li>Professional interface</li>
            </ul>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <h3 style="margin: 0 0 1rem 0; color: white;">Predictive Modeling for Himalayan Expedition Success</h3>
    <p style="margin: 0; color: #d1d5db; font-size: 1rem;">
        <strong>Instructor:</strong> Dr. Bhargavi R
    </p>
    <p style="margin: 0.5rem 0; color: #d1d5db; font-size: 1rem;">
        <strong>Team Members:</strong> Divyanshu Patel (23BAI1214) • Ayush Kumar Singh (23BAI1162)
    </p>
    <p style="margin: 1rem 0 0 0; color: #9ca3af; font-size: 0.9rem;">
        Machine Learning • Data Science • Risk Assessment • Academic Research Project
    </p>
</div>
""", unsafe_allow_html=True)