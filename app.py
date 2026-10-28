from pathlib import Path
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from streamlit_ketcher import st_ketcher
from descriptor_engine import calculate_descriptors as calculate_mixed
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
    # Use genuine Tree SHAP values with a dedicated text column so even
    # very short bars have fully readable contribution labels.
    order = np.argsort(-np.abs(shap_val), kind="stable")
    contributions = shap_val[order]
    names = X_external.columns.to_numpy()[order]
    feature_values = X_external.iloc[0].to_numpy()[order]
    final_probability = base + float(shap_val.sum())
    ends = final_probability - np.r_[0.0, np.cumsum(contributions[:-1])]
    starts = ends - contributions
    y = np.arange(len(order))[::-1]
    colors = np.where(contributions >= 0, "#c2185b", "#006bb6")

    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.32, right=0.82, top=0.84, bottom=0.12)
    ax.barh(y, np.abs(contributions), left=np.minimum(starts, ends),
            height=0.62, color=colors, edgecolor="white", linewidth=0.8)
    for i in range(len(y) - 1):
        ax.plot([starts[i], starts[i]], [y[i]-0.31, y[i+1]+0.31],
                color="#b0b0b0", linestyle=":", linewidth=1)
    labels = [f"{name} = {float(value):.4g}" for name, value in zip(names, feature_values)]
    ax.set_yticks(y, labels=labels, fontsize=12, color="#202020")
    ax.tick_params(axis="y", length=0, pad=12)
    for row, contribution, color in zip(y, contributions, colors):
        ax.text(1.03, row, f"{contribution:+.4f}",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=12, fontweight="bold", color=color, clip_on=False)
    ax.text(1.03, 1.025, "Contribution", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#202020", clip_on=False)
    ax.axvline(base, color="#707070", linestyle="--", linewidth=1, zorder=0)
    ax.axvline(final_probability, color="#303030", linestyle=":", linewidth=1, zorder=0)
    low = min(starts.min(), ends.min(), base)
    high = max(starts.max(), ends.max(), base)
    padding = max((high-low)*0.08, 0.01)
    ax.set_xlim(low-padding, high+padding)
    ax.set_ylim(-0.7, len(order)-0.3)
    ax.tick_params(axis="x", labelsize=11, colors="#202020")
    ax.set_xlabel("Contribution path to Active probability", fontsize=12, labelpad=12)
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#808080")
    fig.suptitle("Contributions to Active probability", fontsize=18, y=0.97)
    fig.text(0.5, 0.915,
             f"Baseline: {base:.4f}     |     Predicted Active probability: {final_probability:.4f}",
             ha="center", fontsize=12, color="#202020")
    fig.text(0.5, 0.035, "Pink increases Active probability; blue decreases it.",
             ha="center", fontsize=11, color="#303030")
    return fig, shap_val

if st.button("Predict activity", type="primary", disabled=not bool(smiles_input.strip())):
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        
        st.subheader("Results")

        try:
            # Calculate descriptors
            with st.spinner("Calculating PaDEL and Mordred descriptors (up to 90 seconds)..."):
                desc_df = calculate_descriptors(smiles_input)
            
            X_external = desc_df
            
            # Make prediction
            y_external_pred = model.predict(X_external)[0]
            y_external_prob = model.predict_proba(X_external)[0]
            
            # Get confidence score
            confidence = y_external_prob[list(model.classes_).index(y_external_pred)]
            
            prediction_done = True

            # Give all 13 descriptors the full page width.
            with st.spinner("Calculating feature contributions..."):
                try:
                    fig, shap_values = create_shap_plot(model, X_external, y_external_pred)
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                except Exception as exc:
                    st.warning(f"SHAP explanation unavailable: {exc}")

            col1, col2 = st.columns(2)
            with col1:
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
