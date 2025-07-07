# sam2_loader.py

import os
import sys
import torch
import numpy as np
import cv2

# Add SAM2 module to path
SAM2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "sam2"))
sys.path.append(SAM2_DIR)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Config and checkpoint paths
HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "checkpoints", "sam2_hiera_large.pt")
CONFIG_PATH = "sam2_hiera_l.yaml"


def load_sam2_model(device="cpu"):
    model = build_sam2(CONFIG_PATH, CHECKPOINT_PATH, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model)
    print("[INFO] SAM2 model loaded.")
    return predictor


def get_segmentation_mask(image_rgb, predictor):
    H, W, _ = image_rgb.shape

    # Define a large circle at the center
    center = np.array([[W // 2, H // 2]])
    # radius = int(min(H, W) * 0.25)  # 25% of smaller dimension
    labels = np.array([1])  # foreground label

    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=center,
        point_labels=labels,
        multimask_output=True
    )

    if masks is None or len(masks) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    return masks[2].astype(np.uint8)
