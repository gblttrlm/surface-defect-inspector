import streamlit as st
import torch
import numpy as np
from PIL import Image
from predict import load_model

st.set_page_config(page_title="Surface Defect Inspector", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def get_model():
    return load_model()


def predict_uploaded_image(image, model, transform, class_names):
    image_gray = image.convert("L")
    x = transform(image_gray).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs


model, transform, class_names = get_model()

st.title("Surface Defect Inspector Demo")
st.write("Upload a surface image and the model will predict the defect class.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png","bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    pred_class, confidence, probs = predict_uploaded_image(
        image, model, transform, class_names
    )

    with col2:
        st.subheader("Prediction Result")
        st.write(f"**Predicted defect:** {pred_class}")
        st.write(f"**Confidence:** {confidence:.2%}")

        # Simple inspection recommendation
        reject_classes = {"scratches", "crazing"}
        if confidence >= 0.80 and pred_class in reject_classes:
            decision = "Reject"
            st.error(f"**Inspection decision:** {decision}")
        elif confidence >= 0.60:
            decision = "Manual Review"
            st.warning(f"**Inspection decision:** {decision}")
        else:
            decision = "Low Confidence - Needs Manual Inspection"
            st.info(f"**Inspection decision:** {decision}")

        st.subheader("Top 3 Predictions")
        top_indices = np.argsort(probs)[::-1][:3]
        for idx in top_indices:
            st.write(f"- {class_names[idx]}: {probs[idx]:.2%}")