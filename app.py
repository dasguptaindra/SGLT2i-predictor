# =========================
# SGLT2i Predictor
# =========================

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------- RDKit & Mordred --------
from rdkit import Chem
from rdkit.Chem import Draw, Lipinski
from mordred import Calculator, descriptors
import shap

# -------- Optional Ketcher --------
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except ImportError:
    KETCHER_AVAILABLE = False

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SGLT2i Predictor",
    page_icon="💊",
    layout="wide"
)

# ================= COMPACT CSS =================
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
h1, h2, h3 { margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ================= MODEL LOADING =================
MODEL_PATH = "gradient_boosting_model_fixed.joblib"
FEATURES_PATH = "model_features.json"

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    model_features = json.load(f)

# Mordred calculator (2D only)
calc = Calculator(descriptors, ignore_3D=True)

# ================= HELPER FUNCTIONS =================
def validate_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def draw_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(300, 300))
    return None


def calculate_descriptors(smiles: str):
    """
    Robust descriptor calculation:
    - Mordred for all descriptors except nHBAcc_Lipinski
    - RDKit Lipinski for nHBAcc_Lipinski
    - Safe handling of failed descriptors
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    mordred_result = calc(mol)

    data = {}
    failed_descriptors = []

    for feat in model_features:
        # RDKit descriptor (explicit)
        if feat == "nHBAcc_Lipinski":
            data[feat] = Lipinski.NumHAcceptors(mol)
            continue

        # Mordred descriptors
        try:
            val = mordred_result[feat]

            if val is None or np.isnan(val) or np.isinf(val):
                raise ValueError("Invalid value")

            data[feat] = float(val)

        except Exception:
            data[feat] = np.nan
            failed_descriptors.append(feat)

    df = pd.DataFrame([data])

    # ---------- SAFE IMPUTATION ----------
    # Median → fallback to zero if entire column fails
    df = df.fillna(df.median()).fillna(0)

    return df, failed_descriptors


# ================= HEADER =================
st.title("SGLT2i Predictor v1.0")

with st.expander("What is SGLT2i Predictor?", expanded=True):
    st.write(
        "**SGLT2i Predictor** allows users to predict the SGLT2 inhibitory activity "
        "of small molecules/drug molecules using a Gradient Boosting classifier "
        "and provides SHAP-based interpretability."
    )

# ================= INPUT SECTION =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("✏️ Draw Molecule")
    smiles_drawn = st_ketcher() if KETCHER_AVAILABLE else ""

with col2:
    st.subheader("🧬 SMILES Input")
    smiles = st.text_input(
        "Enter or edit SMILES",
        value=smiles_drawn if smiles_drawn else ""
    )

# ================= VALIDATION =================
if not smiles:
    st.info("Please draw a molecule or enter a SMILES string.")
    st.stop()

if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES string.")
    st.stop()

# ================= DESCRIPTOR CALCULATION =================
desc_df, failed_desc = calculate_descriptors(smiles)

desc_df = desc_df.replace([np.inf, -np.inf], 0)

if failed_desc:
    st.warning(
        f"⚠️ {len(failed_desc)} descriptor(s) could not be calculated and were imputed:\n"
        f"{', '.join(failed_desc)}"
    )

# ================= PREDICTION =================
pred = model.predict(desc_df)[0]
prob = model.predict_proba(desc_df)[0][1]

# ================= RESULTS =================
st.markdown("---")
st.subheader("📊 Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.image(draw_molecule(smiles), caption="Query Molecule", width=250)

    if pred == 1:
        st.success("🟢 **ACTIVE – SGLT2 Inhibitor**")
    else:
        st.error("🔴 **INACTIVE – Non-Inhibitor**")

    st.metric("Prediction Confidence", f"{prob:.2%}")

with col2:
    st.subheader("📈 SHAP Interpretation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(desc_df)

    # Binary classifier handling
    if isinstance(shap_values, list):
        shap_val = shap_values[1][0]
        base_val = explainer.expected_value[1]
    else:
        shap_val = shap_values[0]
        base_val = explainer.expected_value

    # Select top |SHAP| features
    top_idx = np.argsort(np.abs(shap_val))[-10:]

    fig, ax = plt.subplots(figsize=(5, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_val[top_idx],
            base_values=base_val,
            data=desc_df.iloc[0, top_idx],
            feature_names=desc_df.columns[top_idx]
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

# ================= DESCRIPTORS =================
with st.expander("🔬 Calculated Descriptors"):
    st.dataframe(
        desc_df.T.rename(columns={0: "Value"}),
        use_container_width=True
    )

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<center>🧪 <b>SGLT2i Prediction Tool</b> | Built with Streamlit</center>",
    unsafe_allow_html=True
)
