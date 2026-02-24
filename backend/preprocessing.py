"""
Data transformation pipeline for entrance detection.
Takes raw image bytes, preprocesses for model input.
"""
import numpy as np
import cv2
from typing import Tuple

# Standard input shape for EfficientDet / TF Hub detection models
INPUT_HEIGHT = 320
INPUT_WIDTH = 320


def preprocess_image(image_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    """
    Pipeline: read → resize → normalize → expand dims.
    Returns (batch_tensor, original_width, original_height).
    """
    # Step 1: Read image via OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    # Step 2: Resize to standard input shape (EfficientDet lite0 expects 320x320)
    resized = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # Step 3: EfficientDet lite0 expects uint8 [0,255], not normalized float
    # Step 4: Expand dimensions for batching (batch, height, width, channels)
    batch_tensor = np.expand_dims(resized, axis=0)

    return batch_tensor, orig_w, orig_h
