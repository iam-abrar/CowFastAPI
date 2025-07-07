# depth_utils.py

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEPTH_MODEL_ID = "LiheYoung/depth-anything-large-hf"

def load_depth_model(device="cpu"):
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
    model.eval().to(device)
    print("[INFO] Depth Anything model loaded.")
    return processor, model

def predict_depth(img_rgb, processor, depth_model):
    device = next(depth_model.parameters()).device
    img_pil = Image.fromarray(img_rgb)
    inputs = processor(images=img_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        depth = depth_model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth[None], size=img_rgb.shape[:2], mode="bicubic", align_corners=False
        )[0, 0]
        depth_np = depth.cpu().numpy()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

    return depth_np
