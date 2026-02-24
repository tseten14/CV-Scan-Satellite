"""
Inference and post-processing.
Scales bounding boxes to original image dimensions, filters by confidence.
"""
import time
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf

from preprocessing import preprocess_image, INPUT_HEIGHT, INPUT_WIDTH
from model_loader import load_model

# Show all COCO classes above threshold (entrance-related: person, chair, etc.)
CONFIDENCE_THRESHOLD = 0.40

# COCO 80 class ID -> name (EfficientDet uses COCO)
COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}


def _scale_boxes_to_original(
    boxes: np.ndarray, orig_w: int, orig_h: int
) -> List[Dict[str, int]]:
    """
    Scale normalized [y_min, x_min, y_max, x_max] (0-1) to pixel coords.
    Return list of {xmin, ymin, xmax, ymax}.
    """
    scaled = []
    for box in boxes:
        y_min, x_min, y_max, x_max = box
        xmin = int(round(x_min * orig_w))
        ymin = int(round(y_min * orig_h))
        xmax = int(round(x_max * orig_w))
        ymax = int(round(y_max * orig_h))
        scaled.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return scaled


def _to_title_case(s: str) -> str:
    return s.replace("_", " ").title()


def run_detection(image_bytes: bytes) -> Dict[str, Any]:
    """
    Full pipeline: preprocess → infer → post-process.
    Returns JSON matching frontend DetectionResult.
    """
    start = time.perf_counter()

    # Preprocess
    batch_tensor, orig_w, orig_h = preprocess_image(image_bytes)
    detector_type, detector = load_model()

    # Inference
    if detector_type == "tfhub":
        # EfficientDet lite0: serving_default, expects uint8 images
        input_tensor = tf.convert_to_tensor(batch_tensor, dtype=tf.uint8)
        result = detector(images=input_tensor)
        result = {k: v.numpy() for k, v in result.items()}

        # EfficientDet lite0 outputs: output_0=boxes, output_1=scores, output_2=classes
        boxes = result["output_0"][0]
        scores = result["output_1"][0]
        class_ids = result["output_2"][0]
        class_names = [COCO_CLASSES.get(int(cid), "object") for cid in class_ids]
    else:
        # Custom Keras model - adapt to your model's output format
        preds = detector.predict(batch_tensor)
        # Placeholder: assume output format matches; override if different
        raise NotImplementedError(
            "Custom .h5 model: implement inference for your model's output format. "
            "Place models/entrance_model.h5 with compatible I/O."
        )

    # Post-process: filter by confidence, show all classes
    detections = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, class_names)):
        if score < CONFIDENCE_THRESHOLD:
            continue
        label_str = label.lower() if isinstance(label, str) else str(label).lower()
        scaled_boxes = _scale_boxes_to_original(
            np.array([box]), orig_w, orig_h
        )
        detections.append({
            "id": f"det_{len(detections)}",
            "label": _to_title_case(label_str),
            "confidence": float(score),
            "bbox": scaled_boxes[0],
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "image_width": orig_w,
        "image_height": orig_h,
        "detections": detections,
        "processing_time_ms": elapsed_ms,
    }
