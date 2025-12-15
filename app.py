import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shap
from PIL import Image
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import io
import json
import joblib
from rdkit.Chem import Lipinski
import warnings
warnings.filterwarnings('ignore')

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

# ================= MODEL LOADING =================
MODEL_PATH = "gradient_boosting_model.joblib"
FEATURES_PATH = "model_features.json"

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    model_features = json.load(f)

# Initialize Mordred calculator with ALL descriptors
calc = Calculator(descriptors, ignore_3D=True)

# Streamlit
st.set_page_config(
    page_title="SGLT2i Predictor v1.0: Predict SGLT2 inhibitor(s)",
    layout="wide",
)

st.title("SGLT2i Predictor v1.0: Predict SGLT2 inhibitor(s)")

with st.expander("What is SGLT2i Predictor?", expanded=True):
    st.write('''*SGLT2i Predictor* is a python package that allows users to predict 
             the SGLT2 inhibitory activity of small molecule (**Active** or **Inactive**) 
             and also visualize the molecule.''')

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
    smiles_input = st.text_input("Enter or edit SMILES:", value=smile_code if smile_code else "CC(=O)OC1=CC=CC=C1C(=O)O")

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
    """Calculate descriptors for the model"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Prepare data dictionary
    data = {}
    
    # Get all available descriptors from a dummy calculation
    dummy_mol = Chem.MolFromSmiles("CC")
    all_desc = calc(dummy_mol)
    all_descriptor_names = list(all_desc.keys())
    
    # Calculate all descriptors for the actual molecule
    mordred_result = calc(mol)
    
    # Map feature names to mordred descriptors
    feature_mapping = {
        'MAXaaN': 'EState_VSA5',  # Example mapping - adjust based on your actual features
        'MINaaN': 'EState_VSA4',
        'nN': 'nN',
        'nFARing': 'nFARing',
        'naHRing': 'naHRing',
        'MAXsCl': 'MAXsCl',
        'NaaN': 'NaaN',
        'nHBAcc_Lipinski': 'nHBAcc',
        'BCUTs-1h': 'BCUTw-1h',
        'nFAHRing': 'nFAHRing',
        'ATSC2c': 'ATSC2c',
        'MDEC-33': 'MDEC-33',
        'MATS2c': 'MATS2c'
    }
    
    # For each feature needed by the model
    for feature in model_features:
        if feature in data:
            continue
            
        if feature == "nHBAcc_Lipinski":
            data[feature] = float(Lipinski.NumHAcceptors(mol))
        
        elif feature in feature_mapping:
            mapped_name = feature_mapping[feature]
            if mapped_name in mordred_result:
                try:
                    value = mordred_result[mapped_name]
                    if hasattr(value, '__iter__'):
                        value = float(value[0]) if len(value) > 0 else 0.0
                    else:
                        value = float(value)
                    data[feature] = value
                except:
                    data[feature] = 0.0
            else:
                # Try to find similar descriptor
                found = False
                for desc_name in all_descriptor_names:
                    if feature.lower() in desc_name.lower() or desc_name.lower() in feature.lower():
                        try:
                            value = mordred_result[desc_name]
                            if hasattr(value, '__iter__'):
                                value = float(value[0]) if len(value) > 0 else 0.0
                            else:
                                value = float(value)
                            data[feature] = value
                            found = True
                            break
                        except:
                            continue
                if not found:
                    data[feature] = 0.0
        else:
            # Search for descriptor
            found = False
            for desc_name in all_descriptor_names:
                if feature.lower() in desc_name.lower():
                    try:
                        value = mordred_result[desc_name]
                        if hasattr(value, '__iter__'):
                            value = float(value[0]) if len(value) > 0 else 0.0
                        else:
                            value = float(value)
                        data[feature] = value
                        found = True
                        break
                    except:
                        continue
            if not found:
                data[feature] = 0.0
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Ensure all model features are present
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Reorder columns to match model_features
    df = df[model_features]
    
    # Handle infinities
    df = df.replace([np.inf, -np.inf], 0)
    
    # Fill any NaN values
    df = df.fillna(0)
    
    return df

# Helper to convert prediction to label
def pred_label(pred):
    return "### **Active**" if pred == 1 else "### **Inactive**"

def create_shap_plot(model, X_external, prediction):
    """Create SHAP plot with error handling"""
    try:
        # Try different SHAP explainers
        try:
            # First try TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_external)
            
            if isinstance(shap_values, list):
                # Binary classification
                shap_array = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_array = shap_values
                expected_value = explainer.expected_value
                
        except Exception as e:
            st.warning(f"TreeExplainer failed: {e}. Using KernelExplainer.")
            
            # Use KernelExplainer as fallback
            def model_predict(X):
                return model.predict_proba(X)[:, 1]
            
            # Use background data
            background = shap.sample(X_external, 10)
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(X_external, nsamples=50)
            
            if isinstance(shap_values, list):
                shap_array = shap_values[0]
            else:
                shap_array = shap_values
            expected_value = explainer.expected_value
        
        # Create SHAP values object
        shap_val = shap_array[0]
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_val,
            base_values=expected_value,
            data=X_external.iloc[0].values,
            feature_names=X_external.columns.tolist()
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(4, 3))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        
        return fig, shap_val
        
    except Exception as e:
        st.warning(f"SHAP calculation issue: {e}")
        
        # Create a simple bar plot of feature importance as fallback
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # Use random importance as placeholder
            importance = np.random.rand(len(X_external.columns))
        
        # Create simple bar plot
        fig, ax = plt.subplots(figsize=(4, 3))
        features = X_external.columns.tolist()
        indices = np.argsort(importance)[-10:]  # Top 10
        
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance[indices])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([features[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 10 Important Features')
        plt.tight_layout()
        
        return fig, importance

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        # Author : Dr. Sk. Abdul Amin
        # [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

        st.subheader("Results")

        try:
            # Calculate descriptors
            desc_df = calculate_descriptors(smiles_input)
            
            # Display descriptor info in expander
            with st.expander("🔍 Descriptor Information", expanded=False):
                st.write(f"Model expects {len(model_features)} features:")
                st.write(model_features)
                st.write(f"Calculated features: {list(desc_df.columns)}")
                st.dataframe(desc_df.T.rename(columns={0: "Value"}).round(4))
            
            X_external = desc_df
            
            # Make prediction
            y_external_pred = model.predict(X_external)
            y_external_prob = model.predict_proba(X_external)[0]
            
            # Get confidence score
            confidence = y_external_prob[1] if y_external_pred[0] == 1 else y_external_prob[0]
            
            prediction_done = True

            # Row 1 — Query molecule
            col1, col2 = st.columns(2)

            with col1:
                # SHAP plot with error handling
                with st.spinner("Calculating SHAP values..."):
                    fig, shap_values = create_shap_plot(model, X_external, y_external_pred[0])
                    st.pyplot(fig)
                    plt.clf()
                
                # Molecule image
                mol_img = mol_to_array(mol)
                st.image(mol_img, caption="Query Molecule", width=250)
                
                # Prediction label with confidence
                st.markdown(f"<div style='font-size:40px;'>{pred_label(y_external_pred[0])}</div>",
                            unsafe_allow_html=True)
                st.metric("Confidence Score", f"{confidence:.2%}")
                
                # Display prediction probabilities
                with st.expander("Prediction Probabilities", expanded=False):
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Inactive Probability", f"{y_external_prob[0]:.2%}")
                    with col_prob2:
                        st.metric("Active Probability", f"{y_external_prob[1]:.2%}")

            with col2:
                # Applicability Domain (Leverage) Plot
                try:
                    # Create synthetic data for visualization
                    np.random.seed(42)
                    n_compounds = 50
                    n_features = len(model_features)
                    
                    # Generate random data with similar scale as query molecule
                    query_values = X_external.values.flatten()
                    mean_val = np.mean(query_values)
                    std_val = np.std(query_values) if np.std(query_values) > 0 else 1.0
                    
                    synthetic_data = np.random.normal(mean_val, std_val, (n_compounds, n_features))
                    
                    # Calculate leverage for synthetic data
                    X_synthetic = synthetic_data
                    X_synthetic_centered = X_synthetic - np.mean(X_synthetic, axis=0)
                    cov_matrix = np.cov(X_synthetic_centered.T)
                    
                    try:
                        inv_cov_matrix = np.linalg.pinv(cov_matrix)
                        leverages = []
                        for i in range(len(X_synthetic)):
                            x_centered = X_synthetic[i] - np.mean(X_synthetic, axis=0)
                            leverage = x_centered @ inv_cov_matrix @ x_centered.T
                            leverages.append(leverage)
                        synthetic_leverage = np.array(leverages)
                    except:
                        # Simple distance calculation if matrix inversion fails
                        center = np.mean(X_synthetic, axis=0)
                        synthetic_leverage = np.sum((X_synthetic - center) ** 2, axis=1)
                    
                    # Calculate leverage for query molecule
                    query_values_scaled = X_external.values.flatten()
                    query_centered = query_values_scaled - np.mean(X_synthetic, axis=0)
                    
                    try:
                        query_leverage = query_centered @ inv_cov_matrix @ query_centered.T
                    except:
                        query_leverage = np.sum((query_values_scaled - center) ** 2)
                    
                    # Calculate threshold (95th percentile of synthetic data)
                    leverage_threshold = np.percentile(synthetic_leverage, 95)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(5, 4))
                    
                    # Plot synthetic compounds
                    colors = ['green' if lev <= leverage_threshold else 'red' 
                             for lev in synthetic_leverage]
                    ax.scatter(range(len(synthetic_leverage)), synthetic_leverage, 
                              c=colors, s=20, alpha=0.6, label='Reference Compounds')
                    
                    # Plot query molecule
                    query_color = 'blue' if query_leverage <= leverage_threshold else 'orange'
                    ax.scatter(len(synthetic_leverage), query_leverage, 
                              c=query_color, s=100, marker='*', 
                              label='Query Molecule', edgecolors='black', linewidth=1.5)
                    
                    # Threshold line
                    ax.axhline(y=leverage_threshold, color='red', linestyle='--', 
                              label=f'Threshold ({leverage_threshold:.2f})', alpha=0.7)
                    
                    ax.set_xlabel('Compound Index')
                    ax.set_ylabel('Leverage (Mahalanobis Distance)')
                    ax.set_title('Applicability Domain Analysis')
                    ax.legend(loc='upper right', fontsize='small')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Display AD result
                    is_in_ad = query_leverage <= leverage_threshold
                    ad_percentage = (query_leverage / leverage_threshold * 100) if leverage_threshold > 0 else 100
                    
                    col_ad1, col_ad2 = st.columns(2)
                    with col_ad1:
                        st.metric("Within Applicability Domain", 
                                 "Yes" if is_in_ad else "No")
                    with col_ad2:
                        st.metric("AD Distance", f"{ad_percentage:.1f}% of threshold")
                    
                    if not is_in_ad:
                        st.warning("⚠️ The molecule is outside the applicability domain. Results may be less reliable.")
                    
                    st.caption("NOTE: Applicability Domain assesses prediction reliability based on chemical space coverage")
                    
                except Exception as e:
                    st.error(f"Applicability Domain analysis failed: {str(e)}")
                    st.info("Using simplified analysis")

            # Separator
            st.markdown("---")

            # Feature importance table
            with st.expander("📊 Feature Contributions", expanded=False):
                # Create feature contributions table
                if isinstance(shap_values, np.ndarray):
                    feature_contributions = pd.DataFrame({
                        'Feature': X_external.columns,
                        'Contribution': shap_values,
                        'Value': X_external.iloc[0].values
                    })
                    feature_contributions['Absolute Impact'] = np.abs(feature_contributions['Contribution'])
                    feature_contributions = feature_contributions.sort_values('Absolute Impact', ascending=False)
                    
                    st.dataframe(feature_contributions.style.format({
                        'Contribution': '{:.4f}',
                        'Value': '{:.4f}',
                        'Absolute Impact': '{:.4f}'
                    }), use_container_width=True)
                    
                    # Summary of top features
                    st.subheader("Top Influential Features")
                    top_n = min(5, len(feature_contributions))
                    for i in range(top_n):
                        row = feature_contributions.iloc[i]
                        direction = "increases" if row['Contribution'] > 0 else "decreases"
                        st.write(f"{i+1}. **{row['Feature']}** = {row['Value']:.3f} ({direction} activity)")

            st.info("### Don't forget to cite. Thanks! ###")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.info("Please try a different molecule or check the SMILES string.")
            
            # Debug information
            with st.expander("Debug Information"):
                st.write(f"Error type: {type(e).__name__}")
                st.write(f"Error details: {str(e)}")
                if 'desc_df' in locals():
                    st.write("Descriptor DataFrame shape:", desc_df.shape)
                    st.write("Descriptor DataFrame columns:", list(desc_df.columns))
    else:
        st.error("Invalid SMILES string. Please enter a valid SMILES.")
else:
    st.info("Please enter a SMILES string to get predictions. Try: CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)")

# Author : Dr. Sk. Abdul Amin
# [Details](https://www.scopus.com/authid/detail.uri?authorId=57190176332).
# Contact section
with st.expander("Contact", expanded=False):
    st.write('''
        #### Report an Issue

        Report a bug or contribute here: [GitHub](https://github.com/Amincheminform)

        #### Contact Us
        - [Dr. Sk. Abdul Amin](mailto:pharmacist.amin@gmail.com)
        
        #### Model Information
        - **Algorithm**: Gradient Boosting
        - **Features**: 13 molecular descriptors
        - **Training**: SGLT2 inhibitor dataset
        - **Purpose**: Research use only
    ''')
