from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, shutil, tempfile, base64
import numpy as np
import torch
import cv2

from sam2_loader import load_sam2_model, get_segmentation_mask
from depth_utils import load_depth_model, predict_depth
from feature_utils import extract_mask_features
from model import CowWeightEstimator

# ==================== CONFIG ====================
DEVICE = torch.device("cpu")
# MODEL_PATHS = {
#     "side_all_small": "/mnt/78707D0F707CD57A/CowFastAPI/CowWeightApp/backend/model/best_model_singleALL.pth",
#     "side_all_full": "/mnt/78707D0F707CD57A/CowFastAPI/CowWeightApp/backend/model/best_modelFullSideALL.pth",
#     "rgb_only_small": "/mnt/78707D0F707CD57A/CowFastAPI/CowWeightApp/backend/model/best_model_SideRgbOnly.pth",
#     "rgb_only_full": "/mnt/78707D0F707CD57A/CowFastAPI/CowWeightApp/backend/model/best_modelFullSideRBGOnly.pth"
# }
# colab
MODEL_PATHS = {
    "side_all_small": "/content/CowFastAPI/CowWeightApp/backend/model/best_model_singleALL.pth",
    "side_all_full": "/content/CowFastAPI/CowWeightApp/backend/model/best_modelFullSideALL.pth",
    "rgb_only_small": "/content/CowFastAPI/CowWeightApp/backend/model/best_modelFullSideRBGOnly.pth",
    "rgb_only_full": "/content/CowFastAPI/CowWeightApp/backend/model/best_model_SideRgbOnly.pth"
}

LOADED_MODELS = {}

def get_model(model_name: str):
    if model_name not in LOADED_MODELS:
        model = CowWeightEstimator()
        model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=DEVICE))
        model.eval().to(DEVICE)
        LOADED_MODELS[model_name] = model
    return LOADED_MODELS[model_name]

# ==================== APP SETUP ====================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load external models
DEPTH_PROCESSOR, DEPTH_MODEL = load_depth_model(device=DEVICE)
SAM2_PREDICTOR = load_sam2_model(device=DEVICE)

# ==================== API ENDPOINT ====================
@app.post("/predict/")
async def predict(
    image_file: UploadFile = File(...),
    model: str = Form(...),
    height_in_inch: float = Form(None),
    teeth: int = Form(None)
):
    if model not in MODEL_PATHS:
        return JSONResponse(status_code=400, content={"error": f"Invalid model key: {model}"})

    is_rgb_only = "rgb_only" in model

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded image to disk
        image_path = os.path.join(tmpdir, image_file.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image_file.file, f)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return JSONResponse(status_code=400, content={"error": "Could not read uploaded image."})
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Segment cow
        mask = get_segmentation_mask(image_rgb, SAM2_PREDICTOR)

        # Depth estimation or mock
        if is_rgb_only:
            depth = np.zeros_like(mask, dtype=np.float32)
            mean_depth, std_depth = 0.0, 0.0
            height_in_inch = 0.0
            teeth = 0.0
        else:
            depth = predict_depth(image_rgb, DEPTH_PROCESSOR, DEPTH_MODEL)
            depth_vals = depth[mask > 0]
            mean_depth = float(np.mean(depth_vals)) if len(depth_vals) else 0.0
            std_depth = float(np.std(depth_vals)) if len(depth_vals) else 0.0

        # Feature vector
        mask_feats = extract_mask_features(mask)
        features = [
            mask_feats["area"], mask_feats["width"], mask_feats["height"], mask_feats["aspect_ratio"],
            mean_depth, std_depth, height_in_inch, teeth
        ]

        # Image tensor (always 4 channels)
        if is_rgb_only:
            depth_resized = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
        else:
            depth_img = (depth * 255).astype(np.uint8)
            depth_resized = cv2.resize(depth_img, (image_rgb.shape[1], image_rgb.shape[0]))

        img_4ch = np.concatenate([image_rgb, depth_resized[..., None]], axis=-1)
        img_tensor = torch.tensor(img_4ch.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        stacked_img = img_tensor.unsqueeze(0).to(DEVICE)

        # Tabular tensor
        stacked_tab = torch.tensor([features], dtype=torch.float32).to(DEVICE)

        # RGB-only ablation zeroing
        if is_rgb_only:
            stacked_img[:, 3:, :, :] = 0
            stacked_tab.zero_()

        # Load model and predict
        model_to_use = get_model(model)
        with torch.no_grad():
            pred = model_to_use(stacked_img, stacked_tab).item()

        # Visuals (generate base64 masks)
        mask_vis = (mask * 255).astype(np.uint8)
        _, mask_png = cv2.imencode(".png", mask_vis)

        if is_rgb_only:
            depth_vis = np.zeros_like(mask_vis)
        else:
            depth_vis = (depth * 255).astype(np.uint8)
        _, depth_png = cv2.imencode(".png", depth_vis)

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
