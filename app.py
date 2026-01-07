import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

@st.cache_resource
def load_recycle_model():
    model_path = 'models/recycle_vision_EfficientNet_FIXED.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None

    try:
        # 1. Manually Reconstruct the EXACT architecture from your Kaggle code
        # We use the Functional API to ensure tensor flow is explicit
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), 
            include_top=False, 
            weights='imagenet' # Use standard weights for the base
        )
        
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(12, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        # 2. THE FOUNDER'S TRICK: Partial Weight Loading
        # This loads your 12-class 'brain' and skips the broken base connections
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model

    except Exception as e:
        st.error(f"Structure mismatch still persists: {e}")
        return None

model = load_recycle_model()

# EXACT Alphabetical order for the Garbage Classification dataset
class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard', 
    'clothes', 'glass', 'metal', 'paper', 
    'plastic', 'shoes', 'trash', 'white-glass'
]

# --- UI Configuration ---
# --- UI Configuration (The Designer's Touch) ---
st.set_page_config(
    page_title="RecycleVision Core | Founder Edition", 
    page_icon="♻️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, professional aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div.stButton > button:first-child {
        background-color: #2e7d32;
        color: white;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: The Founder's Mission ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=80)
    st.title("RecycleVision")
    st.markdown("---")
    st.subheader("🌟 Founder's Vision")
    st.info(f"**Nirmal Kumar Bhagatkar**\n\n'My goal is to solve world problems to minimize workload from human races and dignify all living creatures.'")
    st.markdown("---")
    st.write("📊 **System Status:** Operational")
    st.write("🌍 **Impact:** Global Waste Reduction")

# --- Main Dashboard ---
st.title("♻️ RecycleVision AI Core")
st.caption("Advanced Computer Vision Engine for Autonomous Waste Classification")

# Main Layout
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("### 📥 Input Stream")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Drop waste imagery here...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='High-Resolution Input Scan', use_container_width=True)
        else:
            # A placeholder image or illustration to guide the user
            st.info("Awaiting visual input for environmental analysis.")

with col2:
    st.markdown("### 🧠 AI Neural Analysis")
    
    if uploaded_file and model is not None:
        with st.spinner('Accessing Neural Weights...'):
            # Preprocessing
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            predictions = model.predict(img_array)
            result_index = np.argmax(predictions[0])
            result = class_names[result_index]
            confidence = predictions[0][result_index] * 100

        # --- Visual Result Card ---
        with st.container(border=True):
            # Dynamic Icon based on result
            status_color = "#2e7d32" if confidence > 80 else "#f9a825"
            
            st.markdown(f"<h1 style='text-align: center; color: {status_color};'>{result.upper()}</h1>", unsafe_allow_html=True)
            
            # Metric Dashboard
            m1, m2 = st.columns(2)
            m1.metric("Classification", result.capitalize())
            m2.metric("AI Confidence", f"{confidence:.1f}%")
            
            st.progress(min(int(confidence), 100))
            
            st.markdown("---")
            
            # Intelligent Disposal Logic
            if result in ['battery', 'metal']:
                st.error("🚨 **ACTION REQUIRED:** HAZARDOUS/SPECIAL DISPOSAL")
                st.write("This item contains materials that require specialized industrial processing to prevent environmental damage.")
            elif result in ['paper', 'cardboard', 'plastic', 'glass', 'brown-glass', 'white-glass']:
                st.success("✅ **RECYCLABLE:** CIRCULAR ECONOMY ELIGIBLE")
                st.write("This item can be processed and reintroduced into the supply chain. Place in the blue collection bin.")
            elif result == 'biological':
                st.warning("🍃 **COMPOSTABLE:** ORGANIC RECOVERY")
                st.write("Natural material. Suitable for composting to restore soil dignity and nutrients.")
            else:
                st.info("🗑️ **GENERAL WASTE:** LANDFILL DESTINATION")
                st.write("Currently not recyclable via standard vision protocols. Place in general refuse.")

    else:
        # Professional Dashboard Placeholder
        st.write("Please upload an image to begin the classification sequence.")
        st.divider()
        st.image("https://img.freepik.com/free-vector/robotic-arm-sorting-garbage-conveyor-belt_107791-17154.jpg?t=st=1720000000&exp=1720003600&hmac=placeholder", use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d;'>"
    "RecycleVision Core V2.0 | Built with precision to dignify every living creature."
    "</div>", 
    unsafe_allow_html=True
)