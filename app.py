# =========================
# SGLT2i Predictor (FINAL & CORRECT - Case Handling Fix)
# =========================

import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- RDKit --------
from rdkit import Chem
from rdkit.Chem import Draw, Lipinski

# -------- Mordred --------
from mordred import Calculator, descriptors

# -------- SHAP --------
import shap

# -------- Optional Ketcher --------
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except:
    KETCHER_AVAILABLE = False


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SGLT2i Predictor",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
h1, h2, h3 { margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ================= LOAD MODEL & FEATURES =================
MODEL_PATH = "gradient_boosting_model.joblib"
FEATURES_PATH = "model_features.json"

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    model_features = json.load(f)

# Convert to list and clean feature names
model_features = [str(feat).strip() for feat in model_features]
st.session_state['model_features'] = model_features


# ================= INIT MORDED =================
calc = Calculator(descriptors, ignore_3D=True)


# ================= HELPER FUNCTIONS =================
def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit."""
    return Chem.MolFromSmiles(smiles) is not None


def draw_molecule(smiles):
    """Draw molecule from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(280, 280)) if mol else None


def calculate_all_mordred_descriptors(smiles: str) -> dict:
    """Calculate all Mordred descriptors for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    
    mordred_result = calc(mol)
    mordred_dict = {}
    
    for desc_name, value in mordred_result.items():
        desc_name_str = str(desc_name)
        try:
            # Convert to float, handle errors
            if isinstance(value, (int, float)):
                mordred_dict[desc_name_str] = float(value)
            elif hasattr(value, 'real'):
                mordred_dict[desc_name_str] = float(value.real)
            else:
                mordred_dict[desc_name_str] = 0.0
        except:
            mordred_dict[desc_name_str] = 0.0
    
    return mordred_dict


def calculate_rdkit_descriptors(smiles: str) -> dict:
    """Calculate RDKit-specific descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    
    rdkit_dict = {
        "nHBAcc_Lipinski": float(Lipinski.NumHAcceptors(mol)),
    }
    
    return rdkit_dict


def create_feature_mapping(all_descriptors, model_features):
    """Create mapping between model features and actual descriptor names."""
    feature_map = {}
    
    # First pass: exact matches
    for feat in model_features:
        if feat in all_descriptors:
            feature_map[feat] = feat  # Direct match
    
    # Second pass: case-insensitive matches for unmatched features
    # Create lowercase lookup dictionary
    desc_lower = {k.lower(): k for k in all_descriptors.keys()}
    
    for feat in model_features:
        if feat not in feature_map:
            feat_lower = feat.lower()
            if feat_lower in desc_lower:
                feature_map[feat] = desc_lower[feat_lower]
            else:
                # No match found - will use 0.0
                feature_map[feat] = None
    
    return feature_map


def prepare_model_input(smiles: str) -> pd.DataFrame:
    """
    Prepare exact model input matching training features.
    
    Steps:
    1. Calculate all Mordred descriptors
    2. Calculate RDKit descriptors
    3. Create proper mapping between model features and calculated descriptors
    4. Preserve exact feature order and names
    """
    
    # Step 1: Get all descriptors
    mordred_dict = calculate_all_mordred_descriptors(smiles)
    rdkit_dict = calculate_rdkit_descriptors(smiles)
    
    # Combine all available descriptors
    all_descriptors = {**mordred_dict, **rdkit_dict}
    
    # Step 2: Create feature mapping
    feature_map = create_feature_mapping(all_descriptors, model_features)
    
    # Step 3: Build model input using the mapping
    model_input = {}
    
    for feat in model_features:
        mapped_name = feature_map[feat]
        
        if mapped_name is not None and mapped_name in all_descriptors:
            model_input[feat] = all_descriptors[mapped_name]
        else:
            # Feature not available - set to 0
            model_input[feat] = 0.0
    
    # Store mapping for debugging
    st.session_state['feature_map'] = feature_map
    st.session_state['all_descriptors_keys'] = list(all_descriptors.keys())
    
    # Step 4: Create DataFrame with exact column order
    df = pd.DataFrame([model_input])
    
    # Ensure column order matches model_features exactly
    df = df[model_features]
    
    # Handle infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    
    return df


# ================= HEADER =================
st.title("SGLT2i Predictor v1.0")

with st.expander("What is SGLT2i Predictor?", expanded=True):
    st.write(
        "**SGLT2i Predictor** predicts SGLT2 inhibitory activity of drug-like molecules "
        "using a Gradient Boosting classifier with **Mordred + RDKit descriptors** "
        "and **SHAP-based interpretability**."
    )


# ================= INPUT =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("✏️ Draw Molecule")
    if KETCHER_AVAILABLE:
        smiles_drawn = st_ketcher()
    else:
        st.warning("Ketcher not available. Install with: pip install streamlit-ketcher")
        smiles_drawn = ""

with col2:
    st.subheader("🧬 SMILES Input")
    default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    
    # Use drawn molecule if available
    if smiles_drawn and validate_smiles(smiles_drawn):
        smiles = st.text_input("Enter SMILES", value=smiles_drawn)
    else:
        smiles = st.text_input("Enter SMILES", value=default_smiles)


# ================= VALIDATION =================
if not smiles:
    st.warning("⚠️ Please enter a SMILES string")
    st.stop()

if not validate_smiles(smiles):
    st.error("❌ Invalid SMILES string. Please check the format.")
    st.stop()


# ================= PREDICTION =================
st.markdown("---")
st.subheader("📊 Prediction Results")

try:
    # Prepare model input with exact feature matching
    desc_df = prepare_model_input(smiles)
    
    # Verify feature matching
    st.write(f"**Model expects {len(model_features)} features**")
    st.write(f"**Input has {len(desc_df.columns)} features**")
    
    # Check for missing features
    missing_features = [feat for feat in model_features 
                       if feat not in desc_df.columns or desc_df[feat].iloc[0] == 0.0]
    
    if missing_features:
        st.warning(f"⚠️ {len(missing_features)} features missing or zero-valued")
        if st.checkbox("Show missing features"):
            st.write(missing_features[:20])
    
    # Make prediction
    pred = model.predict(desc_df)[0]
    prob = model.predict_proba(desc_df)[0][1]
    
    col1, col2 = st.columns(2)
    
    # -------- Left panel: Molecule and Prediction --------
    with col1:
        img = draw_molecule(smiles)
        if img:
            st.image(img, caption="Query Molecule")
        
        # Display prediction result
        if pred == 1:
            st.success("🟢 **ACTIVE** – SGLT2 Inhibitor")
        else:
            st.error("🔴 **INACTIVE** – Non-Inhibitor")
        
        st.metric("Prediction Probability", f"{prob:.2%}")
        
        # Display feature statistics
        st.write("**Feature Statistics:**")
        non_zero = (desc_df.values != 0).sum()
        st.write(f"- Non-zero features: {non_zero}/{len(model_features)}")
        st.write(f"- Min value: {desc_df.values.min():.4f}")
        st.write(f"- Max value: {desc_df.values.max():.4f}")
        st.write(f"- Mean value: {desc_df.values.mean():.4f}")
    
    # -------- Right panel: SHAP Analysis --------
    with col2:
        st.subheader("📈 SHAP Feature Contribution")
        
        try:
            # Initialize SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(desc_df)
            
            # Handle SHAP values format
            if isinstance(shap_values, list):
                shap_val = shap_values[1][0]  # For class 1
                base_val = explainer.expected_value[1]
            else:
                shap_val = shap_values[0]
                base_val = explainer.expected_value
            
            # Create SHAP explanation
            explanation = shap.Explanation(
                values=shap_val,
                base_values=base_val,
                data=desc_df.iloc[0].values,
                feature_names=desc_df.columns.tolist()
            )
            
            # Plot waterfall chart
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"SHAP analysis failed: {str(e)}")
            st.info("Try with a different molecule or check descriptor calculation.")

except Exception as e:
    st.error(f"❌ Error during prediction: {str(e)}")
    import traceback
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
    st.stop()


# ================= DESCRIPTOR TABLE =================
with st.expander("🔬 Model Input Descriptors (First 20)"):
    st.write(f"**Total descriptors:** {len(desc_df.columns)}")
    
    # Show descriptor summary
    desc_summary = pd.DataFrame({
        'Feature': desc_df.columns,
        'Value': desc_df.iloc[0].values,
        'Type': ['Mordred' if 'nHBAcc_Lipinski' not in col else 'RDKit' 
                for col in desc_df.columns]
    })
    
    st.dataframe(desc_summary.head(20), use_container_width=True)
    
    if len(desc_df.columns) > 20:
        st.write(f"... and {len(desc_df.columns) - 20} more descriptors")


# ================= SIDEBAR =================
with st.sidebar:
    st.header("ℹ️ Model Details")
    st.write(f"""
    **Algorithm:** Gradient Boosting  
    **Descriptor Space:** Mordred (2D) + RDKit  
    **Feature Count:** {len(model_features)}  
    **Interpretability:** SHAP TreeExplainer  
    **Validation:** Exact feature matching
    """)
    
    st.markdown("---")
    st.subheader("💡 Example SMILES")
    examples = {
        "Dapagliflozin": "CCCCCC1=CC(=C(C(=C1)C2C(C(C(O2)CO)O)O)OC3C(C(C(C(O3)CO)O)O)O)O",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Metformin": "CN(C)C(=N)NC(=N)N",
        "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O"
    }
    
    for name, smi in examples.items():
        if st.button(f"{name}", key=f"btn_{name}"):
            st.session_state.smiles = smi
            st.rerun()
    
    st.markdown("---")
    st.subheader("⚠️ Troubleshooting")
    st.write("""
    **Common Issues:**
    1. Check SMILES format
    2. Case sensitivity in feature names
    3. Verify feature names in model_features.json
    4. Mordred calculation may fail for complex molecules
    """)
    
    # Display feature count info
    st.info(f"Model expects {len(model_features)} features")
    
    # Debug option
    if st.checkbox("Show debug info"):
        st.write("**First 10 model features:**")
        st.write(model_features[:10])
        
        # Show feature mapping
        if 'feature_map' in st.session_state:
            st.write("**Feature mapping examples:**")
            mapping_examples = []
            for i, (model_feat, actual_feat) in enumerate(st.session_state['feature_map'].items()):
                if i < 10:
                    if actual_feat:
                        mapping_examples.append(f"{model_feat} → {actual_feat}")
                    else:
                        mapping_examples.append(f"{model_feat} → NOT FOUND")
            st.write(mapping_examples)
        
        # Check for MINaaN specifically
        minaan_variations = ['MINaaN', 'minaaN', 'MinAaN', 'MINAAN']
        found_variation = None
        actual_value = None
        
        for variation in minaan_variations:
            if variation.lower() in [k.lower() for k in st.session_state.get('all_descriptors_keys', [])]:
                # Find the actual key
                actual_key = next((k for k in st.session_state['all_descriptors_keys'] 
                                 if k.lower() == variation.lower()), None)
                if actual_key:
                    found_variation = actual_key
                    # Get the value
                    if 'all_descriptors' in locals():
                        actual_value = all_descriptors.get(actual_key, 'N/A')
                    break
        
        if found_variation:
            st.write(f"**MINaaN variations found:** {found_variation}")
            if actual_value:
                st.write(f"**Value:** {actual_value}")
        
        # Feature match statistics
        matched = sum(1 for feat in model_features 
                     if feat in desc_df.columns and desc_df[feat].iloc[0] != 0.0)
        st.write(f"**Features with non-zero values:** {matched}/{len(model_features)}")


# ================= FOOTER =================
st.markdown("---")
st.caption("SGLT2i Predictor v1.0 | Built with RDKit, Mordred, and Scikit-learn")
