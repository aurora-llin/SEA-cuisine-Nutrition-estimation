import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient
import tomllib
import database as db
from model_utils import load_model, predict_dish

st.title("Southeast Asian Food AI")

@st.cache_resource
def get_model():
    return load_model()

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

    # --- STEP 1: EfficientNet dish classification ---
    st.subheader("Step 1 · Dish Classification")
    model = get_model()
    predictions = predict_dish(model, image, top_k=3)
    top_dish, top_conf = predictions[0]

    for name, conf in predictions:
        st.write(f"{name}: {conf*100:.1f}%")

    dish_id = st.selectbox(
        "Dish (change if prediction is wrong)",
        options=[p[0] for p in predictions],
        index=0
    )

    # --- STEP 2: YOLO ingredient detection (same photo) ---
    st.subheader("Step 2 · Ingredient Detection (YOLO)")
    result = client.infer(image_path, model_id=st.secrets["ROBOFLOW_MODEL_ID"])
    detected = [(p["class"], p["confidence"]) for p in result.get("predictions", [])]
    detected_ids = {d[0] for d in detected}

    for label, conf in detected:
        st.write(f"{label}: {conf*100:.1f}%")

    # --- STEP 3: Merge standard manifest + YOLO detections ---
    manifest = db.get_dish_manifest(dish_id)
    if "ingredients" not in st.session_state or st.session_state.get("last_dish") != dish_id:
        rows = []
        for _, row in manifest.iterrows():
            source = "Recipe + YOLO" if row["ingredient_id"] in detected_ids else "Recipe only"
            rows.append({
                "ingredient_id": row["ingredient_id"],
                "grams": row["typical_quantity_g"],
                "source": source
            })
        st.session_state["ingredients"] = rows
        st.session_state["last_dish"] = dish_id

    # --- STEP 4: User edits ingredient list ---
    st.subheader("Step 4 · Confirm Ingredients")
    to_remove = None
    for i, item in enumerate(st.session_state["ingredients"]):
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        c1.write(item["ingredient_id"])
        item["grams"] = c2.number_input(
            "grams", value=float(item["grams"]), key=f"grams_{i}", label_visibility="collapsed"
        )
        c3.write(item["source"])
        if c4.button("Remove", key=f"remove_{i}"):
            to_remove = i
    if to_remove is not None:
        st.session_state["ingredients"].pop(to_remove)
        st.rerun()

    all_ingredients = db.read_ingredients_nutrition()["ingredient_id"].tolist()
    new_ing = st.selectbox("Choose another ingredient...", [""] + all_ingredients)
    new_grams = st.number_input("Grams to add", value=50.0, key="new_grams")
    if st.button("+ Add ingredient") and new_ing:
        st.session_state["ingredients"].append({
            "ingredient_id": new_ing, "grams": new_grams, "source": "Manual"
        })
        st.rerun()

    # --- STEP 5: Recalculate nutrition ---
    if st.button("Recalculate nutrition"):
        residual = db.compute_hidden_extras(dish_id) or {
            "calories_kcal": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "fiber_g": 0
        }
        totals = dict(residual)
        breakdown = []
        for item in st.session_state["ingredients"]:
            contrib = db.compute_ingredient_contrib(item["ingredient_id"], item["grams"])
            if contrib is None:
                st.warning(f"No nutrition data found for {item['ingredient_id']}")
                continue
            breakdown.append({"ingredient": item["ingredient_id"], "grams": item["grams"], **contrib})
            for k in totals:
                totals[k] += contrib[k]

        st.subheader("Step 5 · Nutrition Estimate")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calories", f"{totals['calories_kcal']:.0f} kcal")
        c2.metric("Protein", f"{totals['protein_g']:.1f} g")
        c3.metric("Fat", f"{totals['fat_g']:.1f} g")
        c4.metric("Carbs", f"{totals['carbs_g']:.1f} g")

        st.write("Per-ingredient breakdown")
        st.dataframe(breakdown)
        st.caption(f"Includes untracked oil/salt/sauce residual: {residual['calories_kcal']:.0f} kcal")
    result = client.infer(image_path, model_id=model_id)

    st.subheader("Raw Roboflow Result")
    st.json(result)
