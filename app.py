import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import timm
import cv2
import os
import importlib
import gdown
from torchvision import transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# --- ENVIRONMENT FIX ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TensorFlow/Keras is only required for the external explainable .h5 model.

# --- CONFIGURATION ---
MODEL_PATH = os.environ.get('MODEL_PATH', 'best_model_mamba_256_final.pth')
EXPLAINABLE_MODEL_PATH = os.environ.get('EXPLAINABLE_MODEL_PATH', 'best_model_final.h5')
IMG_SIZE = (256, 256)

CLASS_LABELS = {
    0: "No DR (Healthy)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR (Critical)"
}

# --- DOWNLOAD MODELS FROM GOOGLE DRIVE ---
@st.cache_resource
def download_models():
    mamba_drive_id = 'YOUR_ACTUAL_MAMBA_ID_HERE'  # <-- Replace with your actual Vision Mamba ID
    h5_drive_id = 'YOUR_ACTUAL_H5_ID_HERE'        # <-- Replace with your actual h5 ID
    
    if not os.path.exists(MODEL_PATH) and mamba_drive_id != 'YOUR_MAMBA_MODEL_DRIVE_ID':
        with st.spinner("Downloading Vision Mamba model (one-time setup)..."):
            gdown.download(id=mamba_drive_id, output=MODEL_PATH, quiet=False)
            
    if not os.path.exists(EXPLAINABLE_MODEL_PATH) and h5_drive_id != 'YOUR_H5_MODEL_DRIVE_ID':
        with st.spinner("Downloading Explainable AI model (one-time setup)..."):
            gdown.download(id=h5_drive_id, output=EXPLAINABLE_MODEL_PATH, quiet=False)

# ==========================================
# 1. VISION MAMBA GRAD-CAM CLASS
# ==========================================
class MambaGradCAM:
    """Extracts heatmaps directly from the Vision Mamba architecture."""
    def __init__(self, model):
        self.model = model
        self.target_layer = model.stages[-1]
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        loss = output[0, class_idx]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.mul(self.activations, weights).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / (heatmap_max + 1e-7)
            
        return heatmap.detach().cpu().numpy().squeeze()

# ==========================================
# 2. LIME WRAPPER FOR KERAS EXPLAINABLE MODEL
# ==========================================
class KerasLIMEWrapper:
    """Wraps the external Keras .h5 model for LIME compatibility."""
    def __init__(self, keras_model, target_size):
        self.model = keras_model
        self.target_size = target_size

    def predict(self, images):
        processed = []
        for img in images:
            img_uint8 = np.uint8(np.clip(img, 0, 255))
            img_resized = cv2.resize(img_uint8, (self.target_size[1], self.target_size[0]))
            img_float = img_resized.astype('float32') / 255.0
            processed.append(img_float)

        batch = np.stack(processed, axis=0)
        return self.model.predict(batch, verbose=0)

@st.cache_resource
def load_explainable_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        st.warning("TensorFlow is not installed in this environment. Explainable .h5 model will not be available.")
        return None, None

    if not os.path.exists(EXPLAINABLE_MODEL_PATH):
        st.warning(f"Explainable .h5 model not found at {EXPLAINABLE_MODEL_PATH}")
        return None, None

    try:
        inp = keras.Input(shape=(128, 128, 3), name='input_layer_1')
        base = keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_tensor=inp,
            alpha=1.0,
            pooling=None,
        )
        base.trainable = False

        x = base.output
        x = keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        x = keras.layers.Dense(256, activation='relu', name='dense')(x)
        x = keras.layers.Dropout(0.5, name='dropout')(x)
        outputs = keras.layers.Dense(5, activation='softmax', name='dense_1')(x)

        model = keras.Model(inputs=inp, outputs=outputs)
        model.load_weights(EXPLAINABLE_MODEL_PATH, by_name=True)

        target_size = (128, 128)
        return model, target_size
    except Exception as e:
        st.warning(f"Could not load explainable .h5 model: {e}")
        return None, None

# ==========================================
# 3. MODEL & IMAGE UTILITIES
# ==========================================
@st.cache_resource
def load_unified_model():
    try:
        model = timm.create_model('mambaout_small', pretrained=False, num_classes=5)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Critical Error: Could not load {MODEL_PATH}. {e}")
        return None

def process_image(uploaded_file):
    if uploaded_file is None: 
        return None, None
    try:
        img = Image.open(uploaded_file).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0), img
    except Exception as e:
        st.error(f"Image Error: {e}")
        return None, None

# ==========================================
# 4. VISUALIZATION UTILITIES
# ==========================================
def create_gradcam_overlay(original_img, heatmap):
    """Create Grad-CAM heatmap overlay"""
    img_np = np.array(original_img.resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_rgb, 0.4, 0)
    return Image.fromarray(overlay)

@st.cache_resource
def get_lime_explainer():
    return lime_image.LimeImageExplainer()

def generate_lime_explanation(original_img, lime_wrapper, explainer):
    """Generate LIME explanation"""
    try:
        img_array = np.array(original_img.resize(lime_wrapper.target_size))
        
        with st.spinner("🔄 Generating LIME explanation... (this takes 30-60 seconds)"):
            explanation = explainer.explain_instance(
                img_array.astype('double'),
                lime_wrapper.predict,
                top_labels=1,
                hide_color=0,
                num_samples=300  # Reduced for speed in Streamlit
            )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        marked_image = mark_boundaries(temp / 255.0, mask, color=(0, 1, 1), mode='subpixel')
        return Image.fromarray((marked_image * 255).astype(np.uint8))
    except Exception as e:
        st.warning(f"LIME generation issue: {e}")
        return None

def generate_clinical_report(results):
    """Generate a formal clinical DR diagnostic report using Streamlit components"""
    from datetime import datetime
    
    # Extract key findings
    left_result = results[0] if len(results) > 0 else None
    right_result = results[1] if len(results) > 1 else None
    max_grade = max([r['grade'] for r in results])
    overall_severity = CLASS_LABELS[max_grade]
    
    # Severity mapping for clinical notes
    severity_map = {
        0: "No retinopathy detected",
        1: "Mild nonproliferative changes observed",
        2: "Moderate nonproliferative diabetic retinopathy",
        3: "Severe nonproliferative diabetic retinopathy",
        4: "Proliferative diabetic retinopathy (critical stage)"
    }
    
    # Recommendation mapping
    recommendation_map = {
        0: "Continue routine annual screening. Maintain blood glucose and blood pressure control.",
        1: "Routine follow-up within 12 months. Reinforce diabetes management.",
        2: "URGENT REFERRAL to ophthalmologist for comprehensive evaluation and treatment planning.",
        3: "URGENT REFERRAL to ophthalmologist. Immediate intervention may be required.",
        4: "EMERGENCY REFERRAL to retina specialist. Immediate evaluation required. Risk of vision loss is high."
    }
    
    # Report Header
    st.markdown("## 📋 DIABETIC RETINOPATHY DIAGNOSTIC REPORT")
    st.markdown(f"**Report Date:** {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    st.markdown(f"**Analysis System:** Vision Mamba DR Diagnostic Portal v1.0")
    
    st.divider()
    
    # Clinical Findings Section
    st.markdown("### 🔍 CLINICAL FINDINGS")
    
    # Overall Assessment
    col_overall = st.columns(1)[0]
    with col_overall:
        if max_grade == 0:
            st.success("✓ **NO DIABETIC RETINOPATHY DETECTED**", icon="✓")
        elif max_grade == 1:
            st.warning("⚠ **MILD DIABETIC RETINOPATHY** (Grade 1)", icon="⚠")
        elif max_grade < 4:
            st.warning(f"⚠ **{overall_severity.upper()}** (Grade {max_grade})", icon="⚠")
        else:
            st.error(f"🚨 **{overall_severity.upper()} - CRITICAL** (Grade {max_grade})", icon="🚨")
        
        st.markdown(f"*{severity_map.get(max_grade, 'Unknown condition')}*")
    
    # Eye-Specific Findings
    st.markdown("#### Eye-Specific Findings")
    
    report_cols = st.columns(2 if (left_result and right_result) else 1)
    
    if left_result:
        with report_cols[0]:
            st.markdown("**👁️ Left Eye (OS)**")
            st.markdown(f"- **Diagnosis:** {left_result['label']}")
            st.markdown(f"- **Confidence:** {left_result['confidence']:.1f}%")
            st.markdown(f"- **Grade:** {left_result['grade']}")
    
    if right_result:
        col_idx = 1 if left_result else 0
        with report_cols[col_idx]:
            st.markdown("**👁️ Right Eye (OD)**")
            st.markdown(f"- **Diagnosis:** {right_result['label']}")
            st.markdown(f"- **Confidence:** {right_result['confidence']:.1f}%")
            st.markdown(f"- **Grade:** {right_result['grade']}")
    
    st.divider()
    
    # Clinical Impression
    st.markdown("### 📌 CLINICAL IMPRESSION")
    st.markdown(f"**Severity Classification:** {overall_severity}")
    st.markdown(f"{severity_map.get(max_grade, 'No assessment available')}")
    
    st.divider()
    
    # Recommendations
    st.markdown("### 💊 RECOMMENDATIONS")
    
    if max_grade >= 4:
        st.error("🚨 **EMERGENCY ACTION REQUIRED**")
        st.markdown(recommendation_map[4])
    elif max_grade >= 3:
        st.error("🚨 **URGENT ACTION REQUIRED**")
        st.markdown(recommendation_map[3])
    elif max_grade >= 2:
        st.warning("⚠️ **SPECIALIST REFERRAL RECOMMENDED**")
        st.markdown(recommendation_map[2])
    elif max_grade >= 1:
        st.info("ℹ️ **ROUTINE FOLLOW-UP RECOMMENDED**")
        st.markdown(recommendation_map[1])
    else:
        st.success("✓ **NO ABNORMALITIES - ROUTINE SCREENING**")
        st.markdown(recommendation_map[0])
    
    st.divider()
    
    # Disclaimer
    st.markdown("### ⚖️ MEDICAL DISCLAIMER")
    disclaimer_text = """
    This report is generated by an AI-assisted diagnostic system and is intended to support clinical decision-making. 
    It should **NOT** replace professional medical evaluation by a qualified ophthalmologist. 
    All findings require clinical correlation and confirmation by licensed eye care professionals.
    
    **System:** Vision Mamba DR Portal | **Method:** Deep Learning with Explainable AI (Grad-CAM + LIME)
    """
    st.info(disclaimer_text)

# ==========================================
# 5. MAIN ANALYSIS LOGIC
# ==========================================
def run_analysis(tensor, original_img, model, cam_tool, lime_wrapper=None, explainer=None):
    # 1. Vision Mamba Prediction
    with torch.set_grad_enabled(True):
        output = model(tensor)
        probs = F.softmax(output[0], dim=0)
        grade = torch.argmax(probs).item()
        confidence = probs[grade].item() * 100

        # 2. Generate Grad-CAM
        heatmap = cam_tool.generate_heatmap(tensor, grade)
        gradcam_img = create_gradcam_overlay(original_img, heatmap)
    
    lime_img = None
    if lime_wrapper is not None and explainer is not None:
        lime_img = generate_lime_explanation(original_img, lime_wrapper, explainer)

    return {
        'grade': grade,
        'confidence': confidence,
        'gradcam': gradcam_img,
        'lime': lime_img,
        'label': CLASS_LABELS[grade]
    }

# ==========================================
# 6. STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(page_title="Vim-DR Portal", layout="wide")
    download_models()
    st.title("👁️ Vision Mamba DR Diagnostic Portal")
    st.markdown("**Explainable AI with Grad-CAM + LIME Interpretability**")

    model = load_unified_model()
    explainable_model, explainable_target_size = load_explainable_model()
    if model:
        cam_tool = MambaGradCAM(model)
        lime_wrapper = None
        explainer = None
        if explainable_model is not None:
            lime_wrapper = KerasLIMEWrapper(explainable_model, explainable_target_size)
            explainer = get_lime_explainer()
        
        # Upload section in a single row
        col1, col2 = st.columns(2)
        with col1:
            left_file = st.file_uploader("Upload Left Eye (OS)", type=["jpg", "png", "jpeg"], key="left")
        with col2:
            right_file = st.file_uploader("Upload Right Eye (OD)", type=["jpg", "png", "jpeg"], key="right")

        if left_file or right_file:
            st.markdown("---")
            results = []
            
            # Process left eye
            if left_file:
                tensor, img = process_image(left_file)
                if tensor is not None and img is not None:
                    res = run_analysis(tensor, img, model, cam_tool, lime_wrapper, explainer)
                    results.append(res)
                    
                    st.subheader("🔍 LEFT EYE (OS) ANALYSIS")
                    
                    # Diagnosis in one row
                    col_diag_l = st.columns(2)
                    with col_diag_l[0]:
                        st.metric("Diagnosis", res['label'])
                    with col_diag_l[1]:
                        st.metric("Confidence", f"{res['confidence']:.1f}%")
                    
                    # Explainability in one row with smaller images
                    col_explain_l = st.columns(2)
                    with col_explain_l[0]:
                        st.markdown("### 📊 Grad-CAM Heatmap")
                        st.markdown("*Shows which regions influenced the prediction*")
                        st.image(res['gradcam'], width=300)
                    
                    with col_explain_l[1]:
                        st.markdown("### 🔬 LIME Explanation")
                        st.markdown("*Local interpretable model-agnostic explanation*")
                        if res['lime']:
                            st.image(res['lime'], width=300)
                        else:
                            st.info("LIME explanation unavailable")
                    
                    # Recommendation
                    if res['grade'] >= 2:
                        st.error("🚨 **REFER TO SPECIALIST (Urgent Action Required)**")
                    else:
                        st.success("✅ **ROUTINE SCREENING (Next check-up: 12 Months)**")

            # Process right eye
            if right_file:
                tensor, img = process_image(right_file)
                if tensor is not None and img is not None:
                    res = run_analysis(tensor, img, model, cam_tool, lime_wrapper, explainer)
                    results.append(res)
                    
                    st.subheader("🔍 RIGHT EYE (OD) ANALYSIS")
                    
                    # Diagnosis in one row
                    col_diag_r = st.columns(2)
                    with col_diag_r[0]:
                        st.metric("Diagnosis", res['label'])
                    with col_diag_r[1]:
                        st.metric("Confidence", f"{res['confidence']:.1f}%")
                    
                    # Explainability in one row with smaller images
                    col_explain_r = st.columns(2)
                    with col_explain_r[0]:
                        st.markdown("### 📊 Grad-CAM Heatmap")
                        st.markdown("*Shows which regions influenced the prediction*")
                        st.image(res['gradcam'], width=300)
                    
                    with col_explain_r[1]:
                        st.markdown("### 🔬 LIME Explanation")
                        st.markdown("*Local interpretable model-agnostic explanation*")
                        if res['lime']:
                            st.image(res['lime'], width=300)
                        else:
                            st.info("LIME explanation unavailable")
                    
                    # Recommendation
                    if res['grade'] >= 2:
                        st.error("🚨 **REFER TO SPECIALIST (Urgent Action Required)**")
                    else:
                        st.success("✅ **ROUTINE SCREENING (Next check-up: 12 Months)**")

            # Final Summary - Clinical Report Format
            if results:
                st.markdown("---")
                generate_clinical_report(results)

    st.sidebar.info(
        "**Model:** Vision Mamba (MambaOut-Small)\n"
        "**XAI Methods:**\n"
        "- Grad-CAM (attention heatmap)\n"
        "- LIME (local interpretability)"
    )

if __name__ == "__main__":
    main()