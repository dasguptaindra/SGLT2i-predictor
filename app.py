# app.py

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# -------------------- RDKit & Mordred Imports --------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Lipinski
except Exception as e:
    st.error(f"RDKit import failed: {e}")
    raise

try:
    from mordred import Calculator, descriptors
except Exception as e:
    st.warning(f"Mordred import warning: {e}")
    Calculator = None
    descriptors = None

try:
    import shap
except Exception as e:
    st.warning(f"SHAP import warning: {e}")
    shap = None

# Optional: Streamlit Ketcher
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except Exception:
    KETCHER_AVAILABLE = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SGLT2 Inhibitor Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    /* Main container styling */
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }
    
    /* Cards */
    .card { background: white; border-radius: 20px; padding: 25px; margin-bottom: 20px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); border-left: 5px solid #667eea; }
    
    /* Headers */
    .main-header { 
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        padding: 10px;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4a5568;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 12px 30px;
        border-radius: 50px;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f7fafc;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    
    .badge-success { background: #48bb78; color: white; }
    .badge-warning { background: #ed8936; color: white; }
    .badge-danger { background: #f56565; color: white; }
    .badge-info { background: #4299e1; color: white; }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- MODEL LOADING --------------------
MODEL_PATH = "gradient_boosting_model_fixed.joblib"
FEATURES_PATH = "model_features.json"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(FEATURES_PATH):
        st.error(f"Features file not found: {FEATURES_PATH}")
        st.stop()
    
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        model_features = json.load(f)
    return model, model_features

model, model_features = load_model()

# -------------------- HELPER FUNCTIONS --------------------
def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string."""
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False

def calculate_selected_descriptors(smiles: str, features: list) -> pd.DataFrame:
    """Calculate selected descriptors using Mordred and RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mordred_dict = {}
    if Calculator and descriptors:
        try:
            calc = Calculator(descriptors, ignore_3D=True)
            mordred_results = calc(mol)
            for k, v in mordred_results.items():
                try:
                    if v is None or str(v) == 'nan' or str(v) == 'NaN':
                        mordred_dict[str(k)] = 0.0
                    else:
                        mordred_dict[str(k)] = float(v)
                except (ValueError, TypeError):
                    mordred_dict[str(k)] = 0.0
        except Exception as e:
            st.warning(f"Mordred calculation failed: {e}")

    rdkit_dict = {}
    try:
        rdkit_dict["nHBAcc_Lipinski"] = float(Lipinski.NumHAcceptors(mol))
    except Exception:
        rdkit_dict["nHBAcc_Lipinski"] = 0.0

    feature_values = {}
    for feat in features:
        if feat in mordred_dict:
            feature_values[feat] = mordred_dict[feat]
        elif feat in rdkit_dict:
            feature_values[feat] = rdkit_dict[feat]
        else:
            try:
                if feat.startswith('n') and len(feat) <= 4:
                    symbol = feat[1:]
                    count = sum(1 for a in mol.GetAtoms() if a.GetSymbol().lower() == symbol.lower())
                    feature_values[feat] = float(count)
                else:
                    feature_values[feat] = 0.0
            except Exception:
                feature_values[feat] = 0.0

    desc_df = pd.DataFrame([feature_values], columns=features)
    desc_df = desc_df.fillna(0.0)
    desc_df = desc_df.replace([np.inf, -np.inf], 0.0)
    
    return desc_df

def clean_dataframe_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has no NaN, None, or inf values before prediction."""
    df_clean = df.copy()
    df_clean = df_clean.fillna(0.0)
    df_clean = df_clean.replace([np.inf, -np.inf], 0.0)
    
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
    
    return df_clean

def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(400, 400))
    return None

def create_gauge_chart(value, title, color_scheme="RdYlGn"):
    """Create a gauge chart for probability display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 30], 'color': '#f56565'},
                {'range': [30, 70], 'color': '#ed8936'},
                {'range': [70, 100], 'color': '#48bb78'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value * 100
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "black", 'family': "Arial"}
    )
    return fig

def create_feature_importance_plot(shap_values, feature_names, top_n=15):
    """Create an interactive feature importance plot."""
    # Sort features by absolute SHAP value
    idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
    sorted_features = [feature_names[i] for i in idx]
    sorted_values = shap_values[idx]
    
    # Create color based on value
    colors = ['#f56565' if v < 0 else '#48bb78' for v in sorted_values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_features,
        x=sorted_values,
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Most Important Features",
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Features",
        height=500,
        template="plotly_white",
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# -------------------- HEADER --------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<div class="main-header">🔬 SGLT2 Inhibitor Predictor</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; font-size: 1.2rem; margin-top: -10px;">Predict SGLT2 inhibition activity using machine learning</p>', unsafe_allow_html=True)

# Decorative separator
st.markdown("""
<div style="height: 3px; background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent); margin: 20px 0;"></div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR INPUT --------------------
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### 🎯 **Input Molecule**')
    st.markdown('Enter your molecule structure to predict SGLT2 inhibition activity')
    
    # Input mode selector with icons
    input_mode = st.radio(
        "**Select Input Method:**",
        ["📝 SMILES String", "🎨 Draw Molecule"],
        index=0
    )
    
    if input_mode == "📝 SMILES String":
        smiles = st.text_area(
            "**SMILES Notation:**",
            height=120,
            placeholder="Enter SMILES here...\n\nExamples:\n• Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O\n• Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            help="Enter the SMILES string of your molecule"
        )
    else:
        if KETCHER_AVAILABLE:
            st.markdown("**Draw your molecule below:**")
            smiles = st_ketcher("", key="ketcher")
        else:
            st.warning("⚠️ **Ketcher not available**")
            st.info("Please install streamlit-ketcher or use SMILES input")
            smiles = st.text_area("**SMILES Notation:**", height=120)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Information card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### ℹ️ **About This Tool**')
    st.markdown("""
    This tool predicts **SGLT2 inhibition activity** using:
    - **Gradient Boosting Model**
    - **Molecular Descriptors** (Mordred & RDKit)
    - **SHAP Explanations**
    
    **Models Used:**
    <span class="badge badge-info">GBM Classifier</span>
    <span class="badge badge-info">200+ Descriptors</span>
    <span class="badge badge-info">SHAP Analysis</span>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

smiles = smiles.strip() if smiles else ""

# -------------------- MAIN CONTENT --------------------
if not smiles:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px;">
            <h1 style="color: #667eea; font-size: 3rem;">🔬</h1>
            <h2 style="color: #4a5568;">Welcome to SGLT2 Predictor</h2>
            <p style="color: #718096; font-size: 1.1rem;">
                👈 **Start by entering a SMILES string** or **drawing a molecule** in the sidebar
            </p>
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 20px; border-radius: 15px; margin-top: 20px;">
                <h3 style="color: white; margin: 0;">💡 Quick Tips</h3>
                <ul style="color: white; text-align: left;">
                    <li>Use valid SMILES notation</li>
                    <li>Try example molecules</li>
                    <li>Check descriptor calculation</li>
                    <li>View SHAP explanations</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if not validate_smiles(smiles):
    st.error("""
    ### ❌ **Invalid SMILES String**
    Please check your input and try again. Ensure:
    - Proper atom symbols
    - Correct bond notation
    - Valid parentheses pairing
    """)
    st.stop()

# -------------------- MOLECULE VISUALIZATION --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 Molecule Visualization</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    mol_img = draw_molecule(smiles)
    if mol_img:
        # Add a decorative border to the molecule image
        st.markdown("""
        <div style="border: 3px solid #667eea; border-radius: 15px; padding: 10px; display: inline-block;">
        """, unsafe_allow_html=True)
        st.image(mol_img, caption="**Molecular Structure**", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Could not generate molecule image")

with col2:
    # Calculate descriptors with progress bar
    st.markdown("### 🧮 **Descriptor Calculation**")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Calculating descriptors... {i+1}%")
    
    desc_df = calculate_selected_descriptors(smiles, model_features)
    
    if desc_df is None:
        st.error("❌ Descriptor calculation failed")
        st.stop()
    
    progress_bar.empty()
    status_text.empty()
    
    # Clean data
    nan_count = desc_df.isna().sum().sum()
    if nan_count > 0:
        desc_df = clean_dataframe_for_prediction(desc_df)
    
    # Display descriptor statistics
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        zero_count = (desc_df.iloc[0] == 0.0).sum()
        st.metric("Zero Values", f"{zero_count}", delta=f"{zero_count/len(model_features)*100:.1f}%")
    
    with col_stat2:
        non_zero = len(model_features) - zero_count
        st.metric("Valid Descriptors", f"{non_zero}", delta="✓" if non_zero > 0 else "⚠️")
    
    with col_stat3:
        max_val = desc_df.max().max()
        min_val = desc_df.min().min()
        st.metric("Value Range", f"{min_val:.2f} - {max_val:.2f}")
    
    # Descriptor preview in tabs
    tab1, tab2 = st.tabs(["📋 Descriptor Values", "📈 Statistics"])
    
    with tab1:
        desc_display = desc_df.T.rename(columns={0: 'Value'})
        desc_display['Status'] = ['✅' if x != 0.0 else '⚠️' for x in desc_display['Value']]
        st.dataframe(desc_display.head(20), use_container_width=True, height=300)
        if len(desc_display) > 20:
            st.caption(f"Showing 20 of {len(desc_display)} descriptors. Use scrollbar to view more.")
    
    with tab2:
        st.dataframe(desc_df.describe(), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION SECTION --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🎯 Make Prediction</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Ready to Predict?
    Click the button below to analyze your molecule's SGLT2 inhibition potential.
    The model will provide:
    - **Activity prediction** (Active/Inactive)
    - **Confidence score**
    - **Detailed explanations**
    """)

with col2:
    predict_button = st.button("🚀 **RUN PREDICTION**", type="primary", use_container_width=True, help="Click to predict SGLT2 inhibition activity")

if predict_button:
    with st.spinner("🤖 **Analyzing molecule...**"):
        # Add prediction animation
        progress_placeholder = st.empty()
        for i in range(101):
            progress_placeholder.progress(i)
        
        # Make prediction
        desc_df_clean = clean_dataframe_for_prediction(desc_df)
        
        if desc_df_clean.isna().any().any():
            st.error("❌ Data cleaning failed. Cannot make prediction.")
            st.stop()
        
        pred = model.predict(desc_df_clean)[0]
        prob = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(desc_df_clean)[0]
            prob = float(probs[1]) if len(probs) == 2 else None
        
        progress_placeholder.empty()
    
    # -------------------- RESULTS DISPLAY --------------------
    st.markdown("## 📊 **Prediction Results**")
    
    # Result cards
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if pred == 1:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">
                <h1 style="font-size: 3rem; margin: 0;">✅</h1>
                <h2 style="margin: 10px 0;">ACTIVE</h2>
                <p>SGLT2 Inhibitor</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);">
                <h1 style="font-size: 3rem; margin: 0;">❌</h1>
                <h2 style="margin: 10px 0;">INACTIVE</h2>
                <p>Not an SGLT2 Inhibitor</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if prob is not None:
            st.plotly_chart(create_gauge_chart(prob, "Confidence Score"), use_container_width=True)
        else:
            st.info("### 📈 Probability Score\nNot available for this model")
    
    with col3:
        # Additional metrics
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h1 style="font-size: 3rem; margin: 0;">🔍</h1>
            <h3>Model Insights</h3>
            <p>Based on {0} descriptors</p>
            <span class="badge badge-info">GBM Model</span>
            <span class="badge badge-info">SHAP Ready</span>
        </div>
        """.format(len(model_features)), unsafe_allow_html=True)
    
    # -------------------- SHAP ANALYSIS --------------------
    if shap:
        st.markdown("---")
        st.markdown('<div class="section-header">📈 Model Interpretation</div>', unsafe_allow_html=True)
        
        with st.spinner("🔍 **Generating feature explanations...**"):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(desc_df_clean)
                expected_value = explainer.expected_value
                
                if isinstance(shap_values, list):
                    shap_val = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                    if shap_val.ndim == 2:
                        shap_val = shap_val[0]
                    base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1 else expected_value
                else:
                    shap_val = shap_values.flatten()
                    base_value = expected_value
                
                # Feature importance plot
                st.plotly_chart(
                    create_feature_importance_plot(shap_val, desc_df_clean.columns, top_n=15),
                    use_container_width=True
                )
                
                # Detailed SHAP values in expander
                with st.expander("📋 **View Detailed Feature Contributions**"):
                    shap_df = pd.DataFrame({
                        'Feature': desc_df_clean.columns,
                        'SHAP Value': shap_val,
                        'Descriptor Value': desc_df_clean.iloc[0].values,
                        'Impact': ['Positive' if v > 0 else 'Negative' for v in shap_val]
                    }).sort_values('SHAP Value', key=abs, ascending=False)
                    
                    st.dataframe(shap_df.head(20), use_container_width=True)
                    
            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")
                st.info("""
                ⚠️ **Alternative Analysis Available**
                While SHAP explanations failed, you can still:
                1. Review the calculated descriptors
                2. Check the confidence score
                3. Validate the SMILES structure
                """)
    
    # -------------------- DOWNLOAD RESULTS --------------------
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # Create results dictionary
        results = {
            "smiles": smiles,
            "prediction": "Active" if pred == 1 else "Inactive",
            "probability": prob if prob else "N/A",
            "descriptors": desc_df_clean.iloc[0].to_dict()
        }
        
        # Convert to JSON for download
        import io
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="📥 **Download Results (JSON)**",
            data=json_str,
            file_name=f"sglt2_prediction_{smiles[:10]}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 **Analyze Another Molecule**", use_container_width=True):
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
<div style="text-align: center; color: #718096; padding: 30px; margin-top: 40px;">
    <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 2px; width: 100px; margin: 0 auto 20px;"></div>
    <h3 style="color: #4a5568;">🔬 SGLT2 Inhibitor Prediction Tool</h3>
    <p>Built with ❤️ using Streamlit • RDKit • Mordred • SHAP</p>
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 15px;">
        <span class="badge badge-info">v1.0.0</span>
        <span class="badge badge-info">ML-Powered</span>
        <span class="badge badge-info">Interactive</span>
    </div>
</div>
""", unsafe_allow_html=True)
