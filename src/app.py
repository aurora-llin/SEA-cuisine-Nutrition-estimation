import hashlib
import os
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tempfile
from html import escape

import streamlit as st
from PIL import Image
from inference_sdk import InferenceHTTPClient

import database as db
from model_utils import load_model, predict_dish

st.set_page_config(
    page_title="Southeast Asian Food AI",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1480px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stFileUploader"] {
            max-width: 900px;
        }

        [data-testid="stMetricValue"] {
            font-size: clamp(1.45rem, 2vw, 2.15rem);
        }

        .confidence-item {
            margin: 0.5rem 0 0.85rem;
        }

        .confidence-label {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.3rem;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .confidence-track {
            width: 100%;
            height: 0.7rem;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(128, 128, 128, 0.2);
        }

        .confidence-fill {
            height: 100%;
            border-radius: inherit;
            transition: width 0.4s ease;
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 0.75rem 2rem;
            }

            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 0.5rem;
            }

            [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Southeast Asian Food AI")
st.caption("Identify a dish, confirm its ingredients, and estimate its nutrition.")


def render_confidence_bar(label, confidence):
    confidence = max(0.0, min(1.0, float(confidence)))
    percentage = confidence * 100
    safe_label = escape(label.replace("_", " ").title())

    if confidence >= 0.75:
        color = "#22c55e"
    elif confidence >= 0.45:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    st.markdown(
        f"""
        <div class="confidence-item" aria-label="{safe_label}: {percentage:.1f}% confidence">
            <div class="confidence-label">
                <span>{safe_label}</span>
                <span>{percentage:.1f}%</span>
            </div>
            <div
                class="confidence-track"
                role="progressbar"
                aria-valuemin="0"
                aria-valuemax="100"
                aria-valuenow="{percentage:.1f}"
            >
                <div
                    class="confidence-fill"
                    style="width: {percentage:.1f}%; background-color: {color};"
                ></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    return load_model()


def get_secret(name):
    try:
        return st.secrets[name]
    except (FileNotFoundError, KeyError):
        return None


roboflow_api_key = get_secret("ROBOFLOW_API_KEY")
roboflow_model_id = get_secret("ROBOFLOW_MODEL_ID")
roboflow_client = (
    InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=roboflow_api_key,
    )
    if roboflow_api_key
    else None
)

uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_path = temp.name

    preview_col, results_col = st.columns([1.15, 1], gap="large")

    with preview_col:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with results_col:
        # --- STEP 1: EfficientNet dish classification ---
        st.subheader("Step 1 · Dish Classification")
        with st.spinner("Classifying dish...", show_time=True):
            model = get_model()
            predictions = predict_dish(model, image, top_k=3)

        for name, conf in predictions:
            render_confidence_bar(name, conf)

        dish_id = st.selectbox(
            "Dish (change if prediction is wrong)",
            options=[prediction[0] for prediction in predictions],
            index=0,
        )

        # --- STEP 2: YOLO ingredient detection (same photo) ---
        st.subheader("Step 2 · Ingredient Detection")
        result = {"predictions": []}
        if roboflow_client is None or not roboflow_model_id:
            st.warning(
                "Ingredient detection is unavailable until ROBOFLOW_API_KEY and "
                "ROBOFLOW_MODEL_ID are added to .streamlit/secrets.toml."
            )
        else:
            try:
                with st.spinner("Detecting ingredients...", show_time=True):
                    result = roboflow_client.infer(
                        image_path,
                        model_id=roboflow_model_id,
                    )
            except Exception as error:
                st.error(f"Ingredient detection failed: {error}")

        try:
            os.unlink(image_path)
        except OSError:
            pass

        detected = [
            (prediction["class"], prediction["confidence"])
            for prediction in result.get("predictions", [])
        ]
        if detected:
            for label, conf in detected:
                render_confidence_bar(label, conf)
        else:
            st.info("No ingredients were detected. You can add them manually below.")

    # --- STEP 3: Merge standard manifest + YOLO detections ---
    with st.spinner("Preparing ingredient list...", show_time=True):
        manifest = db.get_dish_manifest(dish_id)
        all_ingredients = db.read_ingredients_nutrition()["ingredient_id"].tolist()

        def normalize_ingredient_id(value):
            return str(value).strip().lower().replace("-", "_").replace(" ", "_")

        canonical_ingredients = {
            normalize_ingredient_id(ingredient_id): ingredient_id
            for ingredient_id in all_ingredients
        }
        detected_ids = {
            canonical_ingredients.get(normalize_ingredient_id(label), normalize_ingredient_id(label))
            for label, _ in detected
        }
        ingredient_context = (dish_id, image_digest)

        if (
            "ingredients" not in st.session_state
            or st.session_state.get("ingredient_context") != ingredient_context
        ):
            rows = []
            added_ids = set()
            for _, row in manifest.iterrows():
                manifest_id = row["ingredient_id"]
                ingredient_id = canonical_ingredients.get(
                    normalize_ingredient_id(manifest_id),
                    manifest_id,
                )
                source = "Recipe + YOLO" if ingredient_id in detected_ids else "Recipe only"
                rows.append({
                    "ingredient_id": ingredient_id,
                    "grams": row["typical_quantity_g"],
                    "source": source
                })
                added_ids.add(ingredient_id)

            for ingredient_id in sorted(detected_ids - added_ids):
                rows.append({
                    "ingredient_id": ingredient_id,
                    "grams": 50.0,
                    "source": "YOLO only",
                })

            st.session_state["ingredients"] = rows
            st.session_state["ingredient_context"] = ingredient_context

    # --- STEP 3: User edits ingredient list ---
    st.divider()
    st.subheader("Step 3 · Confirm Ingredients")
    st.caption("Adjust the serving amounts, remove incorrect items, or add missing ingredients.")
    to_remove = None
    for i, item in enumerate(st.session_state["ingredients"]):
        c1, c2, c3, c4 = st.columns([3.5, 2, 2.5, 1.25])
        c1.caption("Ingredient")
        c1.write(item["ingredient_id"])
        item["grams"] = c2.number_input(
            "Amount (g)",
            min_value=0.0,
            value=float(item["grams"]),
            step=1.0,
            format="%.1f",
            key=f"grams_{dish_id}_{image_digest[:12]}_{i}",
        )
        c3.caption("Source")
        c3.write(item["source"])
        c4.caption("Action")
        if c4.button(
            "Remove",
            key=f"remove_{dish_id}_{image_digest[:12]}_{i}",
            use_container_width=True,
        ):
            to_remove = i
    if to_remove is not None:
        st.session_state["ingredients"].pop(to_remove)
        st.rerun()

    add_ingredient_col, add_grams_col, add_button_col = st.columns([4, 2, 1.5])
    new_ing = add_ingredient_col.selectbox(
        "Choose another ingredient...",
        [""] + all_ingredients,
    )
    new_grams = add_grams_col.number_input(
        "Amount to add (g)",
        min_value=0.0,
        value=50.0,
        step=1.0,
        format="%.1f",
        key="new_grams",
    )
    add_button_col.write("")
    add_clicked = add_button_col.button(
        "+ Add ingredient",
        use_container_width=True,
    )
    if add_clicked and new_ing:
        st.session_state["ingredients"].append({
            "ingredient_id": new_ing, "grams": new_grams, "source": "Manual"
        })
        st.rerun()

    # --- STEP 4: Recalculate nutrition ---
    st.divider()
    calculate_clicked = st.button(
        "Calculate nutrition",
        type="primary",
        use_container_width=True,
    )
    if calculate_clicked:
        with st.spinner("Calculating nutrition...", show_time=True):
            ingredients = st.session_state["ingredients"]
            ingredient_count = max(len(ingredients), 1)
            progress = st.progress(0, text="Loading recipe nutrition data...")

            residual = db.compute_hidden_extras(dish_id) or {
                "calories_kcal": 0,
                "protein_g": 0,
                "carbs_g": 0,
                "fat_g": 0,
                "fiber_g": 0,
            }
            totals = dict(residual)
            breakdown = []

            for index, item in enumerate(ingredients):
                progress.progress(
                    index / ingredient_count,
                    text=f"Processing {item['ingredient_id']}...",
                )
                contrib = db.compute_ingredient_contrib(
                    item["ingredient_id"],
                    item["grams"],
                )
                if contrib is None:
                    st.warning(f"No nutrition data found for {item['ingredient_id']}")
                    continue
                breakdown.append({
                    "ingredient": item["ingredient_id"],
                    "grams": item["grams"],
                    **contrib,
                })
                for key in totals:
                    totals[key] += contrib[key]

            progress.progress(1.0, text="Nutrition calculation complete.")
            progress.empty()

        st.toast("Nutrition calculation complete", icon="✅")

        st.subheader("Step 4 · Nutrition Estimate")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calories", f"{totals['calories_kcal']:.0f} kcal")
        c2.metric("Protein", f"{totals['protein_g']:.1f} g")
        c3.metric("Fat", f"{totals['fat_g']:.1f} g")
        c4.metric("Carbs", f"{totals['carbs_g']:.1f} g")

        st.write("Per-ingredient breakdown")
        st.dataframe(breakdown, use_container_width=True)
        st.caption(f"Includes untracked oil/salt/sauce residual: {residual['calories_kcal']:.0f} kcal")
