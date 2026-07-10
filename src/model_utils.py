from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "model" / "efficientnet-b0.pth"

CLASS_NAMES = [
    "adobo", "amok_trey", "banh_mi", "hainanese_chicken_rice", "laksa",
    "laphet_thoke", "nasi_goreng", "pad_thai", "pho", "satay",
]

def load_model(weights_path=DEFAULT_WEIGHTS_PATH):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_dish(model, image: Image.Image, top_k=3):
    x = _preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top_probs, top_idx = torch.topk(probs, top_k)
    return [(CLASS_NAMES[i], float(p)) for p, i in zip(top_probs, top_idx)]
