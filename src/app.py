import tempfile
import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient

st.title("SEA Ingredient Detection")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=st.secrets["ROBOFLOW_API_KEY"]
)

uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_path = temp.name

    result = client.infer(
        image_path,
        model_id=st.secrets["ROBOFLOW_MODEL_ID"]
    )

    st.subheader("Raw Roboflow Result")
    st.json(result)