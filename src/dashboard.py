"""
Interactive Dashboard for Heart Disease Analysis
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
from pathlib import Path

# Ensure package root is on path for relative imports if needed
sys.path.append('..')

# Base directory (two levels up from this file: repo root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Page config
st.set_page_config(
    page_title="Heart Disease Analysis Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data_path = BASE_DIR / 'data' / 'processed' / 'heart_disease_cleaned.csv'
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}.\n"
            "Include `data/processed/heart_disease_cleaned.csv` in the repository or provide a valid path."
        )
    return pd.read_csv(data_path)

@st.cache_resource
def load_model():
    model_path = BASE_DIR / 'models' / 'final_best_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}.\n"
            "Ensure `models/final_best_model.pkl` is present in the repository or update the path."
        )
    return joblib.load(model_path)

# Load
df = load_data()
model = load_model()

# Sidebar
st.sidebar.title("❤️ Heart Disease Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📈 Exploratory Analysis", "🤖 Model Insights", "🔮 Prediction Tool"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset**: UCI Heart Disease  
**Records**: 303 patients  
**Features**: 17 attributes  
**Model**: Random Forest (90.16% accuracy)
""")

# Main content
if page == "📊 Overview":
    st.title("📊 Heart Disease Analysis - Overview")
    st.markdown("### Key Metrics Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", f"{len(df)}", "303 records")
    
    with col2:
        disease_rate = df['target'].mean() * 100
        st.metric("Disease Rate", f"{disease_rate:.1f}%", f"{df['target'].sum()} patients")
    
    with col3:
        avg_age = df['age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years", "Range: 29-77")
    
    with col4:
        male_pct = (df['sex']==1).mean() * 100
        st.metric("Male Patients", f"{male_pct:.1f}%", f"{(df['sex']==1).sum()} patients")
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Disease Distribution")
        disease_counts = df['target'].value_counts()
        fig = px.pie(
            values=disease_counts.values,
            names=['No Disease', 'Disease'],
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Age Distribution")
        fig = px.histogram(
            df, x='age', nbins=20,
            color='target',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'target': 'Disease Status', 'age': 'Age (years)'},
            barmode='overlay'
        )
        fig.update_layout(height=400, legend=dict(title="Status", orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Disease by Gender")
        gender_disease = df.groupby(['sex', 'target']).size().reset_index(name='count')
        gender_disease['sex'] = gender_disease['sex'].map({0: 'Female', 1: 'Male'})
        gender_disease['target'] = gender_disease['target'].map({0: 'No Disease', 1: 'Disease'})
        
        fig = px.bar(
            gender_disease, x='sex', y='count', color='target',
            color_discrete_map={'No Disease': '#2ecc71', 'Disease': '#e74c3c'},
            barmode='group'
        )
        fig.update_layout(height=400, xaxis_title="Gender", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Chest Pain Types")
        cp_disease = df.groupby(['cp', 'target']).size().reset_index(name='count')
        cp_disease['target'] = cp_disease['target'].map({0: 'No Disease', 1: 'Disease'})
        
        fig = px.bar(
            cp_disease, x='cp', y='count', color='target',
            color_discrete_map={'No Disease': '#2ecc71', 'Disease': '#e74c3c'},
            barmode='stack'
        )
        fig.update_layout(height=400, xaxis_title="Chest Pain Type (1-4)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Exploratory Analysis":
    st.title("📈 Exploratory Data Analysis")
    
    # Filters
    st.sidebar.markdown("### Filters")
    age_range = st.sidebar.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (30, 70))
    gender_filter = st.sidebar.multiselect("Gender", ['Male', 'Female'], default=['Male', 'Female'])
    
    # Filter data
    df_filtered = df[
        (df['age'] >= age_range[0]) & 
        (df['age'] <= age_range[1])
    ]
    if 'Male' not in gender_filter:
        df_filtered = df_filtered[df_filtered['sex'] != 1]
    if 'Female' not in gender_filter:
        df_filtered = df_filtered[df_filtered['sex'] != 0]
    
    st.markdown(f"**Showing {len(df_filtered)} patients** (filtered from {len(df)} total)")
    
    # Correlation heatmap
    st.markdown("### Correlation Matrix")
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    corr_matrix = df_filtered[features].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Age vs Cholesterol")
        fig = px.scatter(
            df_filtered, x='age', y='chol',
            color='target',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'target': 'Disease'},
            hover_data=['sex', 'cp', 'thalach']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Age vs Max Heart Rate")
        fig = px.scatter(
            df_filtered, x='age', y='thalach',
            color='target',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'target': 'Disease', 'thalach': 'Max Heart Rate'},
            hover_data=['sex', 'cp', 'chol']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plots
    st.markdown("### Feature Distributions by Disease Status")
    
    feature_select = st.selectbox(
        "Select Feature",
        ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    )
    
    fig = px.box(
        df_filtered, x='target', y=feature_select,
        color='target',
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        labels={'target': 'Disease Status'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Insights":
    st.title("🤖 Machine Learning Model Insights")
    
    # Model performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "90.16%", "+1.64% (tuned)")
    
    with col2:
        st.metric("Precision", "81.82%", "High confidence")
    
    with col3:
        st.metric("Recall", "96.43%", "Catches most cases")
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### Feature Importance")
    
    feature_importance = pd.DataFrame({
        'Feature': ['thal', 'thalach', 'cp', 'ca', 'oldpeak', 'age', 'chol', 'trestbps', 'exang', 'slope'],
        'Importance': [0.1442, 0.1182, 0.1172, 0.1057, 0.0974, 0.0820, 0.0819, 0.0760, 0.0419, 0.0410]
    })
    
    fig = px.bar(
        feature_importance.sort_values('Importance', ascending=True),
        x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.markdown("### Model Comparison")
    
    models_data = pd.DataFrame({
        'Model': ['Random Forest', 'Logistic Regression', 'Gradient Boosting', 'SVM', 'Decision Tree'],
        'Accuracy': [0.8852, 0.8689, 0.8525, 0.8197, 0.7049],
        'Precision': [0.8182, 0.8125, 0.7879, 0.7576, 0.6471],
        'Recall': [0.9643, 0.9286, 0.9286, 0.8929, 0.7857],
        'F1-Score': [0.8852, 0.8667, 0.8525, 0.8197, 0.7097]
    })
    
    fig = go.Figure()
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Bar(
            name=metric,
            x=models_data['Model'],
            y=models_data[metric],
            text=models_data[metric].round(3),
            textposition='auto'
        ))
    
    fig.update_layout(
        barmode='group',
        height=400,
        xaxis_title="Model",
        yaxis_title="Score",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    st.markdown("### Confusion Matrix (Random Forest)")
    
    cm_data = [[27, 6], [1, 27]]
    
    fig = px.imshow(
        cm_data,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=['No Disease', 'Disease'],
        y=['No Disease', 'Disease'],
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ True Negatives: 27")
        st.success("✅ True Positives: 27")
    with col2:
        st.error("❌ False Positives: 6")
        st.error("❌ False Negatives: 1")

elif page == "🔮 Prediction Tool":
    st.title("🔮 Heart Disease Prediction Tool")
    st.markdown("Enter patient information to get a prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", 20, 100, 50)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 70, 210, 150)
        exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, 0.1)
    
    with col3:
        slope = st.selectbox("Slope", [1, 2, 3])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [3, 6, 7])
    
    # Engineer features
    age_above_50 = 1 if age > 50 else 0
    high_chol = 1 if chol > 200 else 0
    low_heart_rate = 1 if thalach < 140 else 0
    
    if st.button("🔍 Predict", type="primary"):
        # Create input
        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, 
            oldpeak, slope, ca, thal, age_above_50, high_chol, low_heart_rate
        ]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                     'age_above_50', 'high_chol', 'low_heart_rate'])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        if prediction == 1:
            st.error("⚠️ **HEART DISEASE DETECTED**")
            st.metric("Confidence", f"{probability[1]*100:.2f}%")
            st.warning("**Recommendation**: Consult a cardiologist immediately for further evaluation.")
        else:
            st.success("✅ **NO HEART DISEASE**")
            st.metric("Confidence", f"{probability[0]*100:.2f}%")
            st.info("**Recommendation**: Maintain a healthy lifestyle and regular checkups.")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1]*100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Disease Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prediction==1 else "darkgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("CS 591 - Data Science Project")
st.sidebar.caption("© 2025 - Heart Disease Analysis")