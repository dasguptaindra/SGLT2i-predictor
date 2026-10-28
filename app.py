from pathlib import Path
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from streamlit_ketcher import st_ketcher
from descriptor_engine import calculate_descriptors as calculate_mixed, FEATURE_SOURCES
from PIL import Image
from rdkit import Chem
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


st.title("SGLT2i Predictor v1.0: Predict SGLT2 inhibitor(s)")

with st.expander("What is SGLT2i Predictor?", expanded=True):
    st.write('''**SGLT2i Predictor** is a tool that allows users to predict 
             the SGLT2 inhibitory activity of small molecule (**Potent inhibition** or **Less potent inhibiton**) 
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

@st.cache_data(max_entries=128, show_spinner=False)
def calculate_descriptors(smiles):
    return calculate_mixed(smiles, tuple(model_features))

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

if st.button("Predict activity", type="primary", disabled=not bool(smiles_input.strip())):
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        
        st.subheader("Results")

        try:
            # Calculate descriptors
            with st.spinner("Calculating PaDEL and Mordred descriptors (up to 90 seconds)..."):
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
                display_df["Value"] = display_df["Value"].round(6)
                display_df["Calculator"] = [FEATURE_SOURCES[n] for n in display_df.index]
                display_df["Missing value filled with 0"] = display_df.index.isin(desc_df.attrs.get("zero_filled", []))
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
