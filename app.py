# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Defensive imports
try:
    from rdkit import Chem
    from rdkit.Chem import Lipinski
    from rdkit.Chem.Draw import rdMolDraw2D   # <-- SAFE SVG DRAWING
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

try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except Exception:
    KETCHER_AVAILABLE = False

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="SGLT2 Inhibitor Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- MODEL LOAD --------------------
MODEL_PATH = "random_forest_model.joblib"
FEATURES_PATH = "model_features.json"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error(f"Feature file missing: {FEATURES_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    model_features = json.load(f)

# -------------------- HELPERS --------------------
def validate_smiles(smiles: str) -> bool:
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False

# SAFE HEADLESS SVG DRAWING FOR RDKit
def draw_molecule(smiles: str, w=350, h=350):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace("svg:", "")  # clean namespace

def calculate_selected_descriptors(smiles: str, features: list) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mordred_dict = {}
    if Calculator is not None:
        try:
            calc = Calculator(descriptors, ignore_3D=True)
            mordred_results = calc(mol)
            for k, v in mordred_results.items():
                try:
                    if v is None or str(v) == "nan":
                        mordred_dict[str(k)] = 0.0
                    else:
                        mordred_dict[str(k)] = float(v)
                except Exception:
                    mordred_dict[str(k)] = 0.0
        except:
            mordred_dict = {}

    rdkit_dict = {}
    try:
        rdkit_dict["nHBAcc_Lipinski"] = float(Lipinski.NumHAcceptors(mol))
    except:
        pass

    feature_values = {}
    for feat in features:
        if feat in mordred_dict:
            feature_values[feat] = mordred_dict.get(feat, 0.0)
        elif feat in rdkit_dict:
            feature_values[feat] = rdkit_dict.get(feat, 0.0)
        else:
            try:
                if feat.startswith("n") and len(feat) <= 4:
                    symbol = feat[1:]
                    count = sum(1 for a in mol.GetAtoms() if a.GetSymbol().lower() == symbol.lower())
                    feature_values[feat] = float(count)
                else:
                    feature_values[feat] = 0.0
            except:
                feature_values[feat] = 0.0

    return pd.DataFrame([feature_values], columns=features)

# Manual SHAP fallback waterfall (unchanged)
def create_manual_waterfall(shap_val, base_value, desc_df):
    fig, ax = plt.subplots(figsize=(12, 8))
    features_sorted = sorted(zip(desc_df.columns, shap_val, desc_df.iloc[0].values),
                             key=lambda x: abs(x[1]), reverse=True)[:10]
    features = [f[0] for f in features_sorted]
    values = [f[1] for f in features_sorted]
    ax.bar(range(len(features)), values)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45)
    return fig

# -------------------- UI --------------------
st.title("🧪 SGLT2 Inhibitor Predictor")

st.sidebar.header("Input Molecule")
input_mode = st.sidebar.radio("Select Input Mode", ["SMILES String", "Draw Molecule"])

if input_mode == "SMILES String":
    smiles = st.sidebar.text_area("Enter SMILES", placeholder="Example: C1=CC=CC=C1")
else:
    if KETCHER_AVAILABLE:
        smiles = st_ketcher("", key="ketcher")
    else:
        st.sidebar.warning("Draw mode unavailable. Enter SMILES instead.")
        smiles = st.sidebar.text_area("Enter SMILES")

smiles = smiles.strip()

if not smiles:
    st.stop()

if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES")
    st.stop()

st.subheader("📌 Molecule Structure")
svg = draw_molecule(smiles)
if svg:
    st.image(svg, use_column_width=True)

st.subheader("🧮 Descriptor Values")
desc_df = calculate_selected_descriptors(smiles, model_features)
st.dataframe(desc_df.T, use_container_width=True)

if st.button("🚀 Predict"):
    pred = model.predict(desc_df)[0]
    st.header("Prediction Result")
    st.success("🟢 ACTIVE (SGLT2 Inhibitor)" if pred == 1 else "🔴 INACTIVE")

    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(desc_df)
            st.subheader("Feature Importance (SHAP)")
            shap.summary_plot(shap_values, desc_df, show=False)
            st.pyplot(plt.gcf())
        except:
            fig = create_manual_waterfall(shap_values[0], explainer.expected_value, desc_df)
            st.pyplot(fig)

st.markdown("---")
st.write("Built with ❤️ using Streamlit, RDKit, Mordred & ML")
