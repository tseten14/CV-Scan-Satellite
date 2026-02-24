"""
COCO object detection via EfficientDet lite0 (TF Hub).
Returns bounding boxes for people, vehicles, furniture, etc.
"""
import time
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf

from preprocessing import preprocess_image, INPUT_HEIGHT, INPUT_WIDTH
from model_loader import load_model

CONFIDENCE_THRESHOLD = 0.45

COCO_CLASSES = {
    1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle",
    6: "Bus", 8: "Truck", 9: "Boat", 10: "Traffic Light",
    11: "Fire Hydrant", 13: "Stop Sign", 14: "Parking Meter", 15: "Bench",
    16: "Bird", 17: "Cat", 18: "Dog",
    27: "Backpack", 28: "Umbrella",
    44: "Bottle", 47: "Cup",
    62: "Chair", 63: "Couch", 64: "Potted Plant",
    67: "Dining Table", 72: "Tv", 73: "Laptop",
    85: "Clock", 86: "Vase",
}


def run_detection(image_bytes: bytes) -> Dict[str, Any]:
    """Run EfficientDet COCO detection. Returns DetectionResult dict."""
    start = time.perf_counter()

    batch_tensor, orig_w, orig_h = preprocess_image(image_bytes)
    detector_type, detector = load_model()

    if detector_type != "tfhub":
        raise NotImplementedError("Only TF Hub models are supported")

    input_tensor = tf.convert_to_tensor(batch_tensor, dtype=tf.uint8)
    result = detector(images=input_tensor)
    result = {k: v.numpy() for k, v in result.items()}

    boxes = result["output_0"][0]     # [N, 4] ymin,xmin,ymax,xmax (pixel coords in input space)
    scores = result["output_1"][0]    # [N]
    class_ids = result["output_2"][0] # [N]

    # Scale boxes from input-pixel-space to original image dimensions
    scale_x = orig_w / INPUT_WIDTH
    scale_y = orig_h / INPUT_HEIGHT

    detections: List[Dict[str, Any]] = []
    for box, score, cid in zip(boxes, scores, class_ids):
        if score < CONFIDENCE_THRESHOLD:
            continue
        label = COCO_CLASSES.get(int(cid))
        if label is None:
            continue
        y_min, x_min, y_max, x_max = box
        xmin = max(0, int(round(x_min * scale_x)))
        ymin = max(0, int(round(y_min * scale_y)))
        xmax = min(orig_w, int(round(x_max * scale_x)))
        ymax = min(orig_h, int(round(y_max * scale_y)))
        if xmax <= xmin or ymax <= ymin:
            continue
        detections.append({
            "id": f"coco_{len(detections)}",
            "label": label,
            "confidence": round(float(score), 3),
            "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "image_width": orig_w,
        "image_height": orig_h,
        "detections": detections,
        "processing_time_ms": elapsed_ms,
    }
