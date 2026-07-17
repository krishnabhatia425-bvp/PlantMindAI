import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="PlantMind AI",
    page_icon="🌱",
    layout="centered"
)

st.sidebar.title("🌱 PlantMind AI")

st.sidebar.markdown("""
### 👨‍💻 Developer

Krishna Bhatia

---

### 📌 About

AI Powered Plant Disease Detection System

Built using:

- Python
- TensorFlow
- Streamlit
""")


# Load trained model
import gdown
url = "https://drive.google.com/file/d/18h_JtHJvwY2BPaHB0aJXwsHWNZjnYgEn/view?usp=drivesdk"

gdown.download(url, "plant_disease_model.h5", quiet=False)
print("Model download complete")

model = tf.keras.models.load_model("plant_disease_model.h5")
print("Model loaded successfully")


# App title
st.title("🌱 PlantMind AI")
st.caption("🚀 Smart AI-powered solution for plant health detection")
st.markdown("""
### AI Powered Plant Disease Detection

Upload a leaf image and our AI will analyze whether the plant is **Healthy** or **Diseased**.
""")


# Upload image
uploaded_image = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)
st.info("💡 Tip: Upload a clear leaf image for better prediction accuracy.")


if uploaded_image is not None:

    image = Image.open(uploaded_image)

    st.image(
        image,
        caption="Uploaded plant leaf",
        use_container_width=True
    )


    # Preprocess image
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

# Prediction
    
    with st.spinner("🔍 Analyzing plant image..."):
        prediction = model.predict(img_array)
    confidence = prediction[0][0]

    if confidence > 0.5:
         result = "Diseased Plant 🦠"
         confidence_score = confidence * 100
         st.error(f"🦠 Prediction: {result}")

    else:
        result = "Healthy Plant 🌿"
    confidence_score = (1 - confidence) * 100
    st.success(f"🌿 Prediction: {result}")
    st.balloons()

    st.write(f"Confidence: {confidence_score:.2f}%")
    if confidence_score >= 90:
            st.success("🟢 AI Confidence Level: Excellent")
    elif confidence_score >= 75:
            st.info("🟡 AI Confidence Level: Good")
    else:
            st.warning("🟠 AI Confidence Level: Low (Try uploading a clearer image)")
    st.progress(int(confidence_score))

    if "Diseased" in result:
        st.warning("""
### 🌿 Recommendation

• Remove infected leaves.

• Avoid overwatering.

• Spray a suitable fungicide.

• Keep the plant in a well-ventilated area.
""")
    
    if "Healthy" in result:
        st.success("""
### 🌱 Plant Care Tips

• Continue regular watering.

• Ensure adequate sunlight.

• Use organic fertilizer regularly.

• Inspect leaves weekly for early disease detection.
""")
        
        
st.markdown("---")
st.caption("🌱 PlantMind AI | Developed by Krishna Bhatia | Powered by TensorFlow & Streamlit")