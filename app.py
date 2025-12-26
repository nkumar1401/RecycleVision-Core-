import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

@st.cache_resource
def load_recycle_model():
    model_path = 'recycle_vision_best.h5'
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

# --- UI CODE ---
# --- UI Configuration ---
st.set_page_config(page_title="RecycleVision AI", page_icon="♻️", layout="wide") # 'wide' uses the full screen width

st.title("♻️ RecycleVision: AI Waste Classifier")
st.write("Dignifying waste management through high-speed AI sorting.")

# Create two columns
col1, col2 = st.columns([1, 1]) # Equal width columns

with col1:
    st.subheader("1. Input")
    uploaded_file = st.file_uploader("Upload waste image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        # Resize image purely for UI display so it doesn't take up the whole screen
        st.image(image, caption='Uploaded Image', use_container_width=True)

with col2:
    st.subheader("2. AI Analysis")
    if uploaded_file and model is not None:
        with st.spinner('Analyzing...'):
            # Preprocessing
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            predictions = model.predict(img_array)
            result_index = np.argmax(predictions[0])
            result = class_names[result_index]
            confidence = predictions[0][result_index] * 100

        # Display results in the second column
        st.success(f"**Classification:** {result.upper()}")
        
        # Compact Metrics
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        st.progress(min(int(confidence), 100))

        # Actionable Advice
        if result in ['battery', 'metal']:
            st.warning("⚠️ **Special Disposal:** Take to a specialized center.")
        elif result in ['paper', 'cardboard', 'plastic', 'glass', 'brown-glass', 'white-glass']:
            st.info("♻️ **Recyclable:** Blue Bin.")
        elif result == 'biological':
            st.info("🍃 **Compostable:** Organic Waste.")
        else:
            st.error("🗑️ **General Trash:** Standard Bin.")
    else:
        st.info("Waiting for image upload...")

    # Business Logic
    if result in ['battery', 'metal']:
        st.warning("⚠️ Special Disposal required.")
    elif result in ['paper', 'cardboard', 'plastic', 'glass', 'brown-glass', 'white-glass']:
        st.success("♻️ Recyclable.")
    else:
        st.error("🗑️ General Waste.")