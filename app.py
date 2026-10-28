from pathlib import Path
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from streamlit_ketcher import st_ketcher
from mordred import Calculator, descriptors
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem.Draw import rdMolDraw2D

st.set_page_config(
    page_title="SGLT2i Predictor v1.0: Predict SGLT2 inhibitor(s)",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent / "random_forest_model.joblib"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(str(MODEL_PATH))
    model_features = model.feature_names_in_.tolist()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

calc = Calculator(descriptors, ignore_3D=True)

st.title("SGLT2i Predictor v1.0: Predict SGLT2 inhibitor(s)")

st.warning(
    "Descriptor validation pending: training used both PaDEL and Mordred, "
    "but this app currently uses Mordred and RDKit. The calculator for each "
    "feature and the missing-value treatment must match training before "
    "these SMILES predictions can be relied on."
)

with st.expander("What is SGLT2i Predictor?", expanded=True):
    st.write('''**SGLT2i Predictor** is a tool that allows users to predict 
             the SGLT2 inhibitory activity of small molecule (**Active** or **Inactive**) 
             and also provides SHAP-based interpretability.''')

# Two columns: left for sketcher, right for text input
col1, col2 = st.columns(2)

prediction_done = False

with col1:
    st.markdown("### Draw Query Molecule")
    smile_code = st_ketcher()
    if smile_code and not prediction_done:
        st.success("Molecule drawn successfully!")

with col2:
    st.markdown("### SMILES string of Query Molecule")
    smiles_input = st.text_input("Enter or edit SMILES:", value=smile_code if smile_code else "")

    if smiles_input and not prediction_done:
        st.markdown(f"✅ **SMILES code**: `{smiles_input}`")

def mol_to_array(mol, size=(300, 300)):
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.SetDrawOptions(drawer.drawOptions())
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    return Image.open(io.BytesIO(img_data))

def calculate_descriptors(smiles):
    """Calculate exact descriptor matches; never guess or silently impute."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    values = {str(k): v for k, v in calc(mol).items()}
    # Explicit spelling alias only; verify calculator equivalence against training.
    aliases = {"minaaN": "MINaaN"}
    data = {}
    failures = []
    for feature in model_features:
        name = aliases.get(feature, feature)
        # Retained from the original app. Equivalence to the training
        # nHBAcc_Lipinski definition still requires validation.
        value = Lipinski.NumHAcceptors(mol) if feature == "nHBAcc_Lipinski" else values.get(name)
        try:
            number = float(value)
            if not np.isfinite(number):
                raise ValueError("not finite")
            data[feature] = number
        except (TypeError, ValueError):
            failures.append(feature)
    if failures:
        raise ValueError(
            "Descriptors could not be calculated: " + ", ".join(failures)
            + ". No zeros were substituted. Use the same calculator and "
            "missing-value treatment as the training data."
        )
    return pd.DataFrame([data], columns=model_features)

# Helper to convert prediction to label
def pred_label(pred):
    return "### **🟢 Active**" if pred == 1 else "### **🔴 Inactive**"

def create_shap_plot(model, X_external, prediction):
    """Explain Active (class 1) probability using genuine Tree SHAP values."""
    X_external = X_external.loc[:, model.feature_names_in_]
    class_index = list(model.classes_).index(1)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_external)
    if isinstance(values, list):
        shap_val = np.asarray(values[class_index])[0]
    else:
        values = np.asarray(values)
        if values.ndim != 3:
            raise ValueError(f"Unexpected random forest SHAP shape: {values.shape}")
        shap_val = values[0, :, class_index]
    base = float(np.asarray(explainer.expected_value).reshape(-1)[class_index])
    explanation = shap.Explanation(
        values=shap_val,
        base_values=base,
        data=X_external.iloc[0].to_numpy(),
        feature_names=X_external.columns.tolist(),
    )
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    fig.axes[0].set_title("Contributions to Active probability")
    fig.tight_layout()
    return fig, shap_val

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        
        st.subheader("Results")

        try:
            # Calculate descriptors
            with st.spinner("Calculating molecular descriptors..."):
                desc_df = calculate_descriptors(smiles_input)
            
            # Display descriptor info in expander
            with st.expander("🔍 Descriptor Information", expanded=False):
                st.write(f"**Model expects {len(model_features)} features:**")
                st.write(", ".join(model_features))
                st.write(f"**Calculated features ({len(desc_df.columns)}):**")
                st.write(", ".join(desc_df.columns))
                
                # Show descriptor values
                st.write("**Descriptor Values:**")
                display_df = desc_df.T.copy()
                display_df.columns = ["Value"]
                display_df["Value"] = display_df["Value"].round(4)
                st.dataframe(display_df, width="stretch")
            
            X_external = desc_df
            
            # Make prediction
            y_external_pred = model.predict(X_external)[0]
            y_external_prob = model.predict_proba(X_external)[0]
            
            # Get confidence score
            confidence = y_external_prob[list(model.classes_).index(y_external_pred)]
            
            prediction_done = True

            # Row 1 — Query molecule
            col1, col2 = st.columns(2)

            with col1:
                # SHAP plot
                with st.spinner("Calculating feature contributions..."):
                    try:
                        fig, shap_values = create_shap_plot(model, X_external, y_external_pred)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as exc:
                        st.warning(f"SHAP explanation unavailable: {exc}")
                
                # Molecule image
                mol_img = mol_to_array(mol)
                st.image(mol_img, caption="Query Molecule", width=250)
                
                # Prediction label with confidence
                st.markdown(f"<div style='font-size:40px;'>{pred_label(y_external_pred)}</div>",
                            unsafe_allow_html=True)
                
                # Confidence metrics
                col_conf1, col_conf2 = st.columns(2)
                with col_conf1:
                    st.metric("Predicted-class probability", f"{confidence:.2%}")
                with col_conf2:
                    st.metric("Prediction", "🟢 ACTIVE" if y_external_pred == 1 else "🔴 INACTIVE")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Check the error above; descriptor calculation and model errors are not necessarily invalid SMILES.")
    else:
        st.error("Invalid SMILES string")

# Contact section
with st.expander("Contact & Information", expanded=False):
    st.write('''
        #### 📧 Contact Us      
        #### 🐛 Report Issues
        Report bugs or contribute: [GitHub Repository](https://github.com/dasguptaindra/SGLT2i-predictor)
        
        #### ⚠️ Disclaimer
        This tool is for research purposes only. Not for clinical decision making.
        Always validate predictions with experimental data.
    ''')
