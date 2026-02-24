"""
Global model loading. Load once at server startup.
"""
import os
import tensorflow as tf
import tensorflow_hub as hub

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, "entrance_model.h5")
# SSD MobileNet: fast, well-supported. Or use faster_rcnn for higher accuracy.
TFHUB_MODEL = "https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1"
# Fallback if lite0 unavailable:
TFHUB_FALLBACK = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

# Global model instance, loaded once
_detector = None


def load_model():
    """Load detection model once at startup."""
    global _detector
    if _detector is not None:
        return _detector

    if os.path.exists(CUSTOM_MODEL_PATH):
        model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
        _detector = ("keras", model)
    else:
        try:
            detector = hub.load(TFHUB_MODEL).signatures["serving_default"]
        except Exception:
            detector = hub.load(TFHUB_FALLBACK).signatures.get("serving_default") or hub.load(TFHUB_FALLBACK).signatures["default"]
        _detector = ("tfhub", detector)
    return _detector
