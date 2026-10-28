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
    from matplotlib.patches import Polygon
    from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

    # Eight individual features and one exact sum of the remaining features.
    order = np.argsort(-np.abs(shap_val), kind="stable")
    top = order[:8]
    contributions = list(shap_val[top])
    labels = [(str(X_external.columns[i]), f"{float(X_external.iloc[0, i]):.3f}".rstrip("0").rstrip(".")) for i in top]
    if len(order) > 8:
        contributions.append(float(shap_val[order[8:]].sum()))
        labels.append((f"{len(order)-8} other features", None))
    contributions = np.asarray(contributions)
    final_probability = base + float(shap_val.sum())
    ends = final_probability - np.r_[0.0, np.cumsum(contributions[:-1])]
    starts = ends - contributions
    y = np.arange(len(contributions))[::-1]
    low = min(starts.min(), ends.min(), base)
    high = max(starts.max(), ends.max(), base)
    span = max(high-low, 0.1)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.30, right=0.96, top=0.82, bottom=0.15)
    ax.set_xlim(low-span*0.16, high+span*0.16)
    ax.set_ylim(-0.85, len(y)-0.15)
    ax.set_yticks([])
    ax.grid(False)
    for side in ["top", "left", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=13, length=4, colors="black")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    state = "ACTIVE" if prediction == 1 else "INACTIVE"
    title = f"SHAP Waterfall ({state})"
    ax.set_title(title, fontsize=18, fontweight="bold", pad=48, wrap=True)

    entries = []
    for row, start, end, contribution, (name, value) in zip(y, starts, ends, contributions, labels):
        color = "#ff0051" if contribution >= 0 else "#008bfb"
        left, right = min(start, end), max(start, end)
        tip = min(span*0.012, (right-left)*0.45)
        h = 0.39
        if contribution >= 0:
            vertices = [(left,row-h),(right-tip,row-h),(right,row),(right-tip,row+h),(left,row+h)]
        else:
            vertices = [(right,row-h),(left+tip,row-h),(left,row),(left+tip,row+h),(right,row+h)]
        ax.add_patch(Polygon(vertices, closed=True, facecolor=color, edgecolor="none"))
        parts = []
        if value is not None:
            parts.append(TextArea(f"{value} = ", textprops={"color":"#909090","fontsize":13,"fontweight":"bold"}))
        parts.append(TextArea(name, textprops={"color":"black","fontsize":13,"fontweight":"bold"}))
        box = HPacker(children=parts, align="center", pad=0, sep=0)
        ax.add_artist(AnnotationBbox(box, (0,row), xycoords=ax.get_yaxis_transform(),
                                    xybox=(-10,0), boxcoords="offset points",
                                    box_alignment=(1,0.5), frameon=False, annotation_clip=False))
        text = ax.text((left+right)/2, row, f"{contribution:+.2f}",
                       ha="center", va="center", color="white", fontsize=11,
                       fontweight="bold", clip_on=False)
        entries.append((text,left,right,tip,color,row,contribution))

    # Match the reference's endpoint labels without grid or guide lines.
    ax.text(final_probability, 1.035, f"f(x) = {final_probability:.3f}",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#555555", clip_on=False)
    ax.text(base, -0.105, f"E[f(X)] = {base:.3f}",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=12, fontweight="bold", color="#555555", clip_on=False)
    # Keep numbers inside wide arrows; place them beside short arrows intact.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text, left, right, tip, color, row, contribution in entries:
        available = abs(ax.transData.transform((right-tip, row))[0] - ax.transData.transform((left, row))[0])
        if text.get_window_extent(renderer).width + 12 > available:
            text.set_color(color)
            text.set_position((right+span*0.012 if contribution >= 0 else left-span*0.012, row))
            text.set_ha("left" if contribution >= 0 else "right")
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

            # Render the reference-style waterfall without a descriptor information panel.
            with st.spinner("Calculating feature contributions..."):
                try:
                    fig, shap_values = create_shap_plot(model, X_external, y_external_pred)
                    st.pyplot(fig, width=760)
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
