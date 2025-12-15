# =====================================================
# SGLT2i Predictor (Streamlit App with Correct Descriptor Calculation)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import traceback

from rdkit import Chem
from mordred import Calculator, descriptors
from mordred.error import Error

import shap
from shap import TreeExplainer

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "gradient_boosting_model.joblib"
FEATURE_PATH = "model_features.json"

# =====================================================
# LOAD MODEL & FEATURES
# =====================================================
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r") as f:
        model_features = json.load(f)
except Exception as e:
    st.error(f"Error loading model or feature file: {e}")
    st.stop()

# =====================================================
# DESCRIPTOR NAME MAPPING
# =====================================================
DESCRIPTOR_NAME_MAP = {
    "MINaaN": "MINaaN",
    "MAXaaN": "MAXaaN",
    "BCUTs-1h": "BCUTs-1h",
    "nHBAcc_Lipinski": "nHBAcc",
}

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def calculate_descriptors_from_smiles(smiles: str) -> pd.DataFrame:
    """Calculate Mordred descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    mol = Chem.AddHs(mol)
    calc = Calculator(descriptors, ignore_3D=True)
    desc_series = calc(mol)

    desc_dict = {}
    for name, val in desc_series.items():
        col_name = str(name)
        if isinstance(val, Error):
            desc_dict[col_name] = np.nan
        else:
            try:
                desc_dict[col_name] = float(val)
            except (ValueError, TypeError):
                desc_dict[col_name] = np.nan

    return pd.DataFrame([desc_dict])

def extract_features(desc_df, required_features):
    X = pd.DataFrame(0.0, index=[0], columns=required_features)
    available_columns = [str(c) for c in desc_df.columns]

    for feat in required_features:
        mordred_name = DESCRIPTOR_NAME_MAP.get(feat, feat)
        if mordred_name in available_columns:
            val = desc_df.at[0, mordred_name]
            if pd.notna(val) and val not in [np.inf, -np.inf]:
                X.at[0, feat] = float(val)
            else:
                X.at[0, feat] = 0.0
        else:
            X.at[0, feat] = 0.0
    return X

def print_feature_summary(X, desc_df, model_features):
    available_columns = [str(c) for c in desc_df.columns]
    st.subheader("Feature Value Summary")
    for feat in model_features:
        mordred_name = DESCRIPTOR_NAME_MAP.get(feat, feat)
        value = X.at[0, feat]
        if mordred_name in available_columns:
            original_val = desc_df.at[0, mordred_name]
            status = "✅ Valid" if pd.notna(original_val) and original_val not in [np.inf, -np.inf] else "⚠️ NaN/Inf"
        else:
            status = "⚠️ Not in Mordred"
        st.text(f"{feat:20} ({mordred_name:20}): {value:.6f} {status}")

def explain_prediction(X, model, pred_class):
    try:
        explainer = TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        base_value = explainer.expected_value

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_array = shap_values[pred_class]
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[pred_class])
            else:
                base_value = float(base_value)
        else:
            shap_array = shap_values
            base_value = float(base_value)

        shap_values_to_plot = shap_array[0] if shap_array.ndim == 2 else shap_array
        st.subheader("SHAP Waterfall Plot")
        shap.plots.waterfall(shap_values_to_plot, max_display=12, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
        return True
    except Exception as e:
        st.warning(f"Error generating SHAP explanation: {e}")
        return False

# =====================================================
# STREAMLIT APP LAYOUT
# =====================================================
st.title("🧪 SGLT2i Molecule Predictor")
st.write(
    "Predict SGLT2 inhibitory activity of molecules using a Gradient Boosting model "
    "and explain predictions using SHAP."
)

smiles_input = st.text_input("Enter SMILES string:")

if st.button("Predict"):
    if not smiles_input:
        st.error("SMILES string cannot be empty")
    else:
        try:
            # Descriptor calculation
            desc_df = calculate_descriptors_from_smiles(smiles_input)
            # Feature extraction
            X = extract_features(desc_df, model_features)
            # Feature summary
            print_feature_summary(X, desc_df, model_features)
            # Prediction
            pred_class = model.predict(X)[0]
            pred_proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
            raw_pred = pred_proba[1] if pred_proba is not None else pred_class

            class_names = {0: "Inactive/Negative", 1: "Active/Positive"}
            st.success(f"Prediction: {pred_class} ({class_names.get(pred_class)})")
            if pred_proba is not None:
                st.info(f"Confidence (Active/Positive): {pred_proba[1]:.1%}")

            # SHAP explanation
            explain_prediction(X, model, pred_class)

        except Exception as e:
            st.error(f"Error: {e}")
            st.text(traceback.format_exc())
