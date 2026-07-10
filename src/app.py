import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient
import tomllib

st.title("SEA Ingredient Detection")


def load_secret(name: str, default=None):
    try:
        return st.secrets[name]
    except Exception:
        pass

    candidate_paths = [
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml",
    ]

    for path in candidate_paths:
        if path.exists():
            with path.open("rb") as f:
                data = tomllib.load(f)
            if name in data:
                return data[name]

    return default


api_key = load_secret("ROBOFLOW_API_KEY")
model_id = load_secret("ROBOFLOW_MODEL_ID")

if not api_key or not model_id:
    st.warning(
        "Roboflow secrets are not configured yet. Add them to the project .streamlit/secrets.toml file or src/.streamlit/secrets.toml."
    )
    st.stop()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key,
)

uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_path = temp.name

    result = client.infer(image_path, model_id=model_id)

    st.subheader("Raw Roboflow Result")
    st.json(result)
