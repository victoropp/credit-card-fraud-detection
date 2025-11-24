import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_loader import load_data

# Page Config
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41424C;
        text-align: center;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛡️ FraudGuard AI: Real-time Anomaly Detection")
st.sidebar.markdown("**Created by:** Victor Collins Oppon")
st.sidebar.markdown("**Role:** Data Scientist")
st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style='background-color: #262730; padding: 15px; border-radius: 5px; border: 1px solid #FF4B4B;'>
    <h4 style='margin:0; color: #ffffff;'>System Architecture</h4>
    <p style='margin:0; color: #e0e0e0; font-size: 0.9rem;'>
        This enterprise-grade security system leverages <b>Gradient Boosting (XGBoost)</b> for high-precision fraud detection. 
        It features <b>SHAP-based Explainable AI (XAI)</b> for transparent decision-making and sub-100ms real-time inference latency.
    </p>
</div>
<br>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ℹ️ Technical Specifications")
st.sidebar.info(
    """
    **Core Technology:**
    - **Algorithm:** XGBoost Classifier (Scale_pos_weight optimized)
    - **Interpretability:** SHAP (Game Theoretic Approach)
    - **Serving:** FastAPI (Asynchronous backend)
    
    **Capabilities:**
    - **Precision-Recall Optimization:** Tuned for imbalanced datasets.
    - **Counterfactual Analysis:** Real-time "What-If" simulation.
    """
)

api_url = st.sidebar.text_input("API Endpoint", "http://localhost:8000/predict")

# Load Data - use relative path from project root
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

@st.cache_data
def get_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = get_data()

if df is not None:
    # Main Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔍 Transaction Inspector")
        
        # Transaction Selector
        fraud_indices = df[df['Class'] == 1].index[:20]
        normal_indices = df[df['Class'] == 0].index[:20]
        sample_indices = np.concatenate([fraud_indices, normal_indices])
        
        transaction_id = st.selectbox("Select Transaction ID", sample_indices)
        
        if transaction_id:
            transaction = df.loc[transaction_id]
            true_label = "Fraud" if transaction['Class'] == 1 else "Normal"
            
            # Display Key Metrics
            st.markdown(f"""
            <div class="metric-card">
                <h3>Amount</h3>
                <h2>${transaction['Amount']:.2f}</h2>
                <p>True Label: <b>{true_label}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🛠️ What-If Analysis")
            st.info("Adjust feature values to see how the model reacts.")
            
            with st.expander("ℹ️ Why these features?"):
                st.markdown("""
                **Feature Selection Rationale:**
                These sliders control the **top 3 most important features** used by the XGBoost model:
                - **V14**: The single most critical predictor (Importance: ~59%).
                - **V4 & V11**: Highly influential features in distinguishing fraud.
                
                **Goal:** By modifying these values, you can perform *counterfactual analysis*—testing if a slight change in transaction patterns would have alerted or evaded the system.
                """)
            
            # Sliders for top features (V14, V4, V11 are usually important)
            v14 = st.slider("V14 (Feature)", float(df['V14'].min()), float(df['V14'].max()), float(transaction['V14']))
            v4 = st.slider("V4 (Feature)", float(df['V4'].min()), float(df['V4'].max()), float(transaction['V4']))
            v11 = st.slider("V11 (Feature)", float(df['V11'].min()), float(df['V11'].max()), float(transaction['V11']))
            
            # Update transaction with slider values
            current_features = transaction.drop('Class').values.tolist()
            # Find indices of V14, V4, V11
            cols = df.columns.drop('Class').tolist()
            idx_v14 = cols.index('V14')
            idx_v4 = cols.index('V4')
            idx_v11 = cols.index('V11')
            
            current_features[idx_v14] = v14
            current_features[idx_v4] = v4
            current_features[idx_v11] = v11
            
            if st.button("Analyze Transaction"):
                with st.spinner("Analyzing with AI Model..."):
                    try:
                        response = requests.post(api_url, json={"features": current_features})
                        if response.status_code == 200:
                            result = response.json()
                            prob = result['fraud_probability']
                            is_fraud = result['is_fraud']
                            shap_values = result.get('shap_values', [])
                            
                            # Store result in session state to persist
                            st.session_state['result'] = result
                            st.session_state['features'] = current_features
                        else:
                            st.error(f"API Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

    with col2:
        st.subheader("📊 Real-time Analysis Results")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            prob = result['fraud_probability']
            shap_values = result.get('shap_values', [])
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF4B4B" if prob > 0.5 else "#00CC96"},
                    'steps': [
                        {'range': [0, 50], 'color': "#262730"},
                        {'range': [50, 100], 'color': "#262730"}],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 50}}))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Waterfall (Simulated with Bar Chart)
            if shap_values:
                st.markdown("### 🧠 Model Explanation (SHAP Values)")
                st.write("Which features contributed most to this prediction?")
                
                # Create DataFrame for Plotly
                feature_names = df.columns.drop('Class').tolist()
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values
                })
                
                # Sort by absolute value
                shap_df['Abs Value'] = shap_df['SHAP Value'].abs()
                shap_df = shap_df.sort_values('Abs Value', ascending=False).head(10)
                
                # Color bars by sign
                shap_df['Color'] = shap_df['SHAP Value'].apply(lambda x: '#FF4B4B' if x > 0 else '#00CC96')
                
                fig_shap = px.bar(
                    shap_df, 
                    x='SHAP Value', 
                    y='Feature', 
                    orientation='h',
                    title="Top 10 Influential Features",
                    text_auto='.3f'
                )
                fig_shap.update_traces(marker_color=shap_df['Color'])
                fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
                
                st.plotly_chart(fig_shap, use_container_width=True)
                
                st.info("Positive values (Red) push the model towards 'Fraud', negative values (Green) push towards 'Normal'.")
        else:
            st.info("👈 Select a transaction and click 'Analyze Transaction' to see results.")

    # --- Model Performance Section ---
    st.markdown("---")
    st.subheader("📈 Model Performance & Validation")
    
    # Load metrics
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')
    
    if os.path.exists(METRICS_PATH):
        import json
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        test_metrics = metrics['Test']
        
        # Display Metrics with Explanations
        st.markdown("### Key Performance Indicators (Test Set)")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("ROC-AUC", f"{test_metrics['ROC AUC']:.4f}")
            st.caption("Ability to distinguish between Fraud and Normal.")
            
        with m2:
            st.metric("PR-AUC", f"{test_metrics['PR AUC']:.4f}")
            st.caption("Precision-Recall Area. **Critical for imbalanced data**.")
            
        with m3:
            st.metric("F1 Score", f"{test_metrics['F1 Score']:.4f}")
            st.caption("Harmonic mean of Precision and Recall.")
            
        with m4:
            st.metric("Accuracy", f"{test_metrics['Accuracy']:.4f}")
            st.caption("Overall correctness (can be misleading for imbalance).")
            
        st.info("""
        **Data Scientist Note:** 
        In fraud detection, **PR-AUC (Precision-Recall AUC)** is the gold standard metric. 
        A high ROC-AUC can be achieved even with poor performance on the minority class (Fraud), 
        but a high PR-AUC ensures we are catching frauds (Recall) without too many false alarms (Precision).
        """)
        
        # Display Charts
        st.markdown("### 📊 Validation Curves")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Precision-Recall Curve", "Training Loss"])
        
        with tab1:
            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                st.image(os.path.join(RESULTS_DIR, 'test_confusion_matrix.png'), caption="Test Confusion Matrix", use_container_width=True)
            with col_cm2:
                st.markdown("""
                **Interpretation:**
                - **True Negatives (Top-Left):** Legitimate transactions correctly identified.
                - **False Positives (Top-Right):** Legitimate transactions flagged as fraud (Customer friction).
                - **False Negatives (Bottom-Left):** Fraud missed (Financial loss).
                - **True Positives (Bottom-Right):** Fraud caught.
                """)
                
        with tab2:
            col_pr1, col_pr2 = st.columns(2)
            with col_pr1:
                st.image(os.path.join(RESULTS_DIR, 'test_pr_curve.png'), caption="Test PR Curve", use_container_width=True)
            with col_pr2:
                st.markdown("""
                **Why this matters:**
                The curve shows the trade-off between Precision and Recall for different thresholds.
                - **Top-Right** is better.
                - We aim to maximize Recall (catch fraud) while maintaining acceptable Precision.
                """)
                
        with tab3:
            st.image(os.path.join(RESULTS_DIR, 'training_loss.png'), caption="XGBoost Training History", use_container_width=True)
            st.caption("Shows the model converging without overfitting (Train and Test loss decrease together).")
            
    else:
        st.warning("Metrics not found. Please run `src/generate_metrics.py` first.")

else:
    st.warning("Data not found. Please check the path.")
