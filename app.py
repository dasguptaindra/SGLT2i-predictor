# app.py

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- RDKit & Mordred --------------------
from rdkit import Chem
from rdkit.Chem import Draw, Lipinski

try:
    from mordred import Calculator, descriptors
except Exception:
    Calculator, descriptors = None, None

try:
    import shap
except Exception:
    shap = None

# Optional Ketcher
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except Exception:
    KETCHER_AVAILABLE = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SGLT2 Inhibitor Predictor",
    page_icon="🧪",
    layout="wide"
)

# -------------------- STYLISH CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main-header {
    font-size: 3.2rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg,#1f77b4,#6f42c1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-header {
    text-align: center;
    font-size: 1.4rem;
    color: #555;
    margin-bottom: 2rem;
}

.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.input-area {
    background: linear-gradient(135deg,#f8f9ff,#eef2ff);
    padding: 25px;
    border-radius: 20px;
}

.prediction-box {
    padding: 28px;
    border-radius: 18px;
    font-size: 1.4rem;
    font-weight: 600;
    text-align: center;
}

.active {
    background: linear-gradient(135deg,#c7f0db,#a8e6cf);
    border: 2px solid #28a745;
}

.inactive {
    background: linear-gradient(135deg,#f9c5c5,#f5b7b1);
    border: 2px solid #dc3545;
}

.stButton>button {
    background: linear-gradient(90deg,#ff416c,#ff4b2b);
    color: white;
    font-size: 1.2rem;
    font-weight: 700;
    padding: 14px;
    border-radius: 14px;
    border: none;
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(255,75,43,0.4);
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
MODEL_PATH = "gradient_boosting_model_fixed.joblib"
FEATURES_PATH = "model_features.json"

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    model_features = json.load(f)

# -------------------- FUNCTIONS --------------------
def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def calculate_descriptors(smiles, features):
    mol = Chem.MolFromSmiles(smiles)
    data = {}

    if Calculator and descriptors:
        calc = Calculator(descriptors, ignore_3D=True)
        res = calc(mol)
        for k, v in res.items():
            try:
                data[str(k)] = float(v)
            except Exception:
                data[str(k)] = 0.0

    data["nHBAcc_Lipinski"] = float(Lipinski.NumHAcceptors(mol))

    df = pd.DataFrame([{f: data.get(f, 0.0) for f in features}])
    return df.fillna(0).replace([np.inf, -np.inf], 0)

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(420, 420))

def manual_waterfall(shap_vals, base_value, df):
    top = np.argsort(np.abs(shap_vals))[-10:]
    features = df.columns[top]
    values = shap_vals[top]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["red" if v < 0 else "green" for v in values]
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black")
    ax.set_title("Top Feature Contributions (SHAP Waterfall)")
    plt.tight_layout()
    return fig

# -------------------- HEADER --------------------
st.markdown("<div class='main-header'>🧪 SGLT2 Inhibitor Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Gradient Boosting + SHAP Explainability</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- SIDEBAR --------------------
st.sidebar.header("🔬 Input Molecule")

mode = st.sidebar.radio("Input Method", ["SMILES String", "Draw Molecule"])

if mode == "SMILES String":
    smiles = st.sidebar.text_area("SMILES", height=120)
else:
    smiles = st_ketcher("") if KETCHER_AVAILABLE else st.sidebar.text_area("SMILES", height=120)

smiles = smiles.strip()
if not smiles or not validate_smiles(smiles):
    st.info("👈 Enter a valid SMILES")
    st.stop()

# -------------------- INPUT DISPLAY --------------------
st.markdown("<div class='input-area'>", unsafe_allow_html=True)
c1, c2 = st.columns([1, 2])

with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧬 Molecule")
    st.image(draw_molecule(smiles), use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧮 Descriptors")
    desc_df = calculate_descriptors(smiles, model_features)
    st.dataframe(desc_df.T.rename(columns={0: "Value"}), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
st.markdown("---")
if st.button("🚀 PREDICT SGLT2 ACTIVITY", use_container_width=True):
    pred = model.predict(desc_df)[0]
    prob = model.predict_proba(desc_df)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        cls = "active" if pred == 1 else "inactive"
        txt = "🟢 ACTIVE (Inhibitor)" if pred == 1 else "🔴 INACTIVE"
        st.markdown(f"<div class='prediction-box {cls}'>{txt}</div>", unsafe_allow_html=True)

    with col2:
        st.metric("Confidence", f"{prob:.1%}")

    # -------------------- SHAP WATERFALL --------------------
    if shap:
        st.markdown("---")
        st.subheader("📊 SHAP Feature Contribution")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(desc_df)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
                base_val = explainer.expected_value[1]
            else:
                shap_vals = shap_values[0]
                base_val = explainer.expected_value

            try:
                explanation = shap.Explanation(
                    values=shap_vals,
                    base_values=base_val,
                    data=desc_df.iloc[0],
                    feature_names=desc_df.columns
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig)
                plt.close()
            except Exception:
                st.warning("Default SHAP plot failed, using manual waterfall.")
                fig = manual_waterfall(shap_vals, base_val, desc_df)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP failed: {e}")

# -------------------- FOOTER --------------------
st.markdown("""
<div style='text-align:center;color:#777;margin-top:2rem'>
🧪 <b>SGLT2 Inhibitor Predictor</b><br>
Gradient Boosting • RDKit • Mordred • SHAP
</div>
""", unsafe_allow_html=True)
