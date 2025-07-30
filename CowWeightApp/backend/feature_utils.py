# feature_utils.py

import numpy as np
import torch

import cv2


def extract_mask_features(mask):
    """Compute area, width, height, and aspect ratio from binary mask."""
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        return dict(area=0, width=0, height=0, aspect_ratio=0)
    width = int(x.max() - x.min() + 1)
    height = int(y.max() - y.min() + 1)
    area = int(mask.sum())
    aspect_ratio = float(width / height) if height > 0 else 0.0
    return dict(area=area, width=width, height=height, aspect_ratio=aspect_ratio)


def prepare_input_tensor(image_rgb, depth_map, features, device):
    """Prepare 4-channel image tensor and tabular tensor from inputs."""
    # Normalize and stack image + depth
    depth_uint8 = (depth_map * 255).astype(np.uint8)
    depth_resized = depth_uint8
    if depth_map.shape[:2] != image_rgb.shape[:2]:
        depth_resized = cv2.resize(depth_uint8, (image_rgb.shape[1], image_rgb.shape[0]))

    img_4ch = np.concatenate([image_rgb, depth_resized[..., None]], axis=-1)
    img_tensor = torch.tensor(img_4ch.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    tab_tensor = torch.tensor([features], dtype=torch.float32).to(device)
    return img_tensor, tab_tensor
