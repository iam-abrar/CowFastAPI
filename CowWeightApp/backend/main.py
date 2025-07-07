# main.py (FastAPI backend)
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import zipfile, shutil, os, tempfile, base64
import numpy as np
import torch
from PIL import Image
import cv2

from sam2_loader import load_sam2_model, get_segmentation_mask
from depth_utils import load_depth_model, predict_depth
from feature_utils import extract_mask_features
from model import CowWeightEstimator

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEPTH_PROCESSOR, DEPTH_MODEL = load_depth_model(device=DEVICE)
SAM2_PREDICTOR = load_sam2_model(device=DEVICE)
WEIGHT_MODEL = CowWeightEstimator()
WEIGHT_MODEL.load_state_dict(torch.load("/mnt/78707D0F707CD57A/CowFastAPI/CowWeightApp/backend/model/full_ablation_best_model.pth", map_location=DEVICE))
WEIGHT_MODEL.eval().to(DEVICE)

@app.post("/predict/")
async def predict(
    image_file: UploadFile = File(...),
    height_in_inch: float = Form(...),
    teeth: int = Form(...)
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded image
        image_path = os.path.join(tmpdir, image_file.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image_file.file, f)

        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Segmentation
        mask = get_segmentation_mask(image_rgb, SAM2_PREDICTOR)

        # Depth
        depth = predict_depth(image_rgb, DEPTH_PROCESSOR, DEPTH_MODEL)

        # Extract features
        mask_feats = extract_mask_features(mask)
        depth_vals = depth[mask > 0]
        mean_depth = float(np.mean(depth_vals)) if len(depth_vals) else 0.0
        std_depth = float(np.std(depth_vals)) if len(depth_vals) else 0.0

        features = [
            mask_feats['area'], mask_feats['width'], mask_feats['height'], mask_feats['aspect_ratio'],
            mean_depth, std_depth, height_in_inch, teeth
        ]

        # Format image tensor
        depth_norm = (depth * 255).astype(np.uint8)
        depth_resized = cv2.resize(depth_norm, (image_rgb.shape[1], image_rgb.shape[0]))
        img_4ch = np.concatenate([image_rgb, depth_resized[..., None]], axis=-1)
        img_tensor = torch.tensor(img_4ch.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        # Predict
        stacked_img = img_tensor.unsqueeze(0).to(DEVICE)
        stacked_tab = torch.tensor([features], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred = WEIGHT_MODEL(stacked_img, stacked_tab).item()

        # Encode mask and depth
        _, mask_png = cv2.imencode('.png', (mask * 255).astype(np.uint8))
        _, depth_png = cv2.imencode('.png', (depth * 255).astype(np.uint8))

        return JSONResponse({
            "cow_weight_pred": round(pred, 2),
            "image_count": 1,
            "features": [{
                "filename": image_file.filename,
                "features": features,
                "weight_pred": round(pred, 2),
                "mask_b64": base64.b64encode(mask_png).decode(),
                "depth_b64": base64.b64encode(depth_png).decode()
            }]
        })
