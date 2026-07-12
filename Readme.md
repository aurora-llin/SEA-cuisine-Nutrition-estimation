# Southeast Asian Cuisine Nutrition Estimation

A Streamlit app that identifies a Southeast Asian dish from an uploaded image, detects likely ingredients, lets the user correct serving amounts, and estimates nutrition.

## How it works

1. EfficientNet classifies the dish and shows its top three predictions.
2. Roboflow YOLO optionally detects ingredients in the same image.
3. The user confirms the ingredients and their weights.
4. Nutrition is calculated from recipe and ingredient data stored in Google Sheets.

Supported dishes include adobo, amok trey, banh mi, Hainanese chicken rice, laksa, laphet thoke, nasi goreng, pad Thai, pho, and satay.

## Requirements

- Python 3.10-3.12
- Internet access for nutrition data and Roboflow detection
- The included model file at `model/efficientnet-b0.pth`
- A Roboflow API key and model ID for ingredient detection (optional)

## Setup and run

Run these PowerShell commands from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

New-Item -ItemType Directory -Force .streamlit
Copy-Item src\.streamlit\secrets.example.toml .streamlit\secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
ROBOFLOW_API_KEY = "your-api-key"
ROBOFLOW_MODEL_ID = "workspace/project/version"
```

Then start the app:

```powershell
streamlit run src/app.py
```

Open the local URL printed by Streamlit, upload a JPG or PNG image, confirm the detected ingredients, and select **Calculate nutrition**.

> [!NOTE]
> Roboflow settings are optional; without them, dish classification and manual ingredient entry still work. Nutrition values are estimates and should not be treated as medical advice.

## Project structure

- `src/app.py` - Streamlit interface and workflow
- `src/model_utils.py` - EfficientNet model loading and prediction
- `src/database.py` - Google Sheets nutrition data and calculations
- `model/efficientnet-b0.pth` - trained dish-classification weights
- `requirements.txt` - Python dependencies
