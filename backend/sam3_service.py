
"""
SAM 3 (Segment Anything Model 3) detection service.
Uses Hugging Face transformers for promptable concept segmentation.
"""
import os
import io
import time
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("uvicorn.error")

# Door-focused prompts for street view mode
STREETVIEW_PROMPTS = [
    "door",
    "revolving door",
]

# Building-focused prompts for satellite/aerial view mode
SATELLITE_PROMPTS = [
    "building",
    "roof",
    "house",
]

# Process one prompt at a time to keep RAM low on laptops
_BATCH_SIZE = 1
# Max dimension for inference — larger images are downscaled to reduce compute and RAM
_MAX_INFER_DIM = 768

_model: Any = None
_processor: Any = None
_device: str = "cpu"
_dtype: Any = None


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_sam3() -> bool:
    """Load SAM 3 model and processor. Returns True on success."""
    global _model, _processor, _device, _dtype
    if _model is not None:
        return True

    try:
        import torch
        from transformers import Sam3Model, Sam3Processor

        # Run on CPU to avoid MPS memory spikes that crash laptops
        _device = "cpu"
        _dtype = torch.float32

        logger.info(f"Loading SAM 3 on {_device}…")

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token

        _processor = Sam3Processor.from_pretrained(
            "facebook/sam3",
            token=token,
        )
        _model = Sam3Model.from_pretrained(
            "facebook/sam3",
            token=token,
        ).to(_device)
        _model.eval()

        logger.info("SAM 3 ready.")
        return True
    except Exception as e:
        logger.error(f"SAM 3 failed to load: {e}")
        return False


def _iou(box_a: dict, box_b: dict) -> float:
    """Compute IoU between two boxes (xmin, ymin, xmax, ymax)."""
    ix1 = max(box_a["xmin"], box_b["xmin"])
    iy1 = max(box_a["ymin"], box_b["ymin"])
    ix2 = min(box_a["xmax"], box_b["xmax"])
    iy2 = min(box_a["ymax"], box_b["ymax"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (box_a["xmax"] - box_a["xmin"]) * (box_a["ymax"] - box_a["ymin"])
    area_b = (box_b["xmax"] - box_b["xmin"]) * (box_b["ymax"] - box_b["ymin"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Non-maximum suppression by confidence. Stricter for same-class to remove duplicates."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep: list[dict] = []
    for det in sorted_dets:
        keep_it = True
        for k in keep:
            iou = _iou(det["bbox"], k["bbox"])
            thresh = iou_threshold
            # Road and sidewalk are adjacent - require higher overlap to suppress
            if {det["label"], k["label"]} <= {"road", "sidewalk"}:
                thresh = 0.85
            # Same class (e.g. car/car) - use stricter IoU to remove duplicates
            elif det["label"] == k["label"]:
                thresh = 0.35
            if iou > thresh:
                keep_it = False
                break
        if keep_it:
            keep.append(det)
    return keep


def _overlap_ratio(box_a: dict, box_b: dict) -> float:
    """Intersection over box_a area (how much of A is covered by B)."""
    ix1 = max(box_a["xmin"], box_b["xmin"])
    iy1 = max(box_a["ymin"], box_b["ymin"])
    ix2 = min(box_a["xmax"], box_b["xmax"])
    iy2 = min(box_a["ymax"], box_b["ymax"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (box_a["xmax"] - box_a["xmin"]) * (box_a["ymax"] - box_a["ymin"])
    return inter / area_a if area_a > 0 else 0.0


def _filter_person_building_overlap(detections: list[dict], img_w: int, img_h: int) -> list[dict]:
    """Remove person detections that incorrectly include building (large overlap with building)."""
    buildings = [d for d in detections if d["label"] == "building"]
    persons = [d for d in detections if d["label"] == "person"]
    others = [d for d in detections if d["label"] not in ("building", "person")]

    img_area = img_w * img_h
    filtered_persons: list[dict] = []
    for p in persons:
        p_area = (p["bbox"]["xmax"] - p["bbox"]["xmin"]) * (p["bbox"]["ymax"] - p["bbox"]["ymin"])
        # Remove if person bbox is unrealistically large (>18% of image - likely misdetected building)
        if p_area > 0.18 * img_area:
            continue
        # Only remove if person is almost entirely inside a building (>70% overlap = likely false positive)
        overlap_any = any(_overlap_ratio(p["bbox"], b["bbox"]) > 0.70 for b in buildings)
        if overlap_any:
            continue
        filtered_persons.append(p)

    return buildings + filtered_persons + others


def _filter_google_map_signs(detections: list[dict]) -> list[dict]:
    """Remove sign detections that are Google Street View nav arrows (on road surface)."""
    roads = [d for d in detections if d["label"] == "road"]
    signs = [d for d in detections if d["label"] == "sign"]
    others = [d for d in detections if d["label"] not in ("sign", "road")]

    filtered_signs: list[dict] = []
    for s in signs:
        # Skip signs on road surface - Google nav arrows are overlaid on pavement
        if any(_overlap_ratio(s["bbox"], r["bbox"]) > 0.25 for r in roads):
            continue
        filtered_signs.append(s)

    return roads + filtered_signs + others


def _filter_sign_pole_on_building(detections: list[dict]) -> list[dict]:
    """Remove sign/pole detections that overlap buildings (likely false positives)."""
    buildings = [d for d in detections if d["label"] == "building"]
    signs = [d for d in detections if d["label"] == "sign"]
    poles = [d for d in detections if d["label"] == "pole"]
    others = [d for d in detections if d["label"] not in ("sign", "pole", "building")]

    def keep_det(d: dict) -> bool:
        return not any(_overlap_ratio(d["bbox"], b["bbox"]) > 0.35 for b in buildings)

    filtered_signs = [s for s in signs if keep_det(s)]
    filtered_poles = [p for p in poles if keep_det(p)]

    return others + buildings + filtered_signs + filtered_poles


def _filter_car_doors(detections: list[dict]) -> list[dict]:
    """Remove door detections that overlap cars or trucks (car doors, not building entrances)."""
    vehicles = [d for d in detections if d["label"] in ("car", "truck")]
    doors = [d for d in detections if d["label"] == "door"]
    others = [d for d in detections if d["label"] not in ("door", "car", "truck")]

    filtered_doors: list[dict] = []
    for door in doors:
        # Skip doors that overlap significantly with a vehicle (car doors)
        if any(_overlap_ratio(door["bbox"], v["bbox"]) > 0.4 for v in vehicles):
            continue
        filtered_doors.append(door)

    return others + vehicles + filtered_doors


def _merge_sidewalk_detections(detections: list[dict], img_h: int) -> list[dict]:
    """Keep at most 2 sidewalk detections (largest by area)."""
    sidewalks = [d for d in detections if d["label"] == "sidewalk"]
    others = [d for d in detections if d["label"] != "sidewalk"]

    if len(sidewalks) <= 2:
        return detections

    def bbox_area(d: dict) -> float:
        b = d["bbox"]
        return (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])

    # Keep only the 1 largest sidewalk region
    sorted_sw = sorted(sidewalks, key=bbox_area, reverse=True)
    return others + sorted_sw[:1]


# Max detections per class - keeps only highest-confidence to reduce noise
_MAX_PER_CLASS: dict[str, int] = {
    "road": 1,
    "sidewalk": 1,
    "building": 50,
    "roof": 50,
    "house": 50,
    "door": 3,
    "car": 5,
    "truck": 2,
    "person": 2,
    "bicycle": 2,
    "tree": 4,
    "vegetation": 2,
    "grass": 2,
    "pole": 4,
    "sign": 3,
    "street light": 3,
    "trash can": 1,
    "bench": 1,
    "fire hydrant": 1,
    "mailbox": 1,
    "traffic light": 2,
    "bus": 2,
    "motorcycle": 1,
}


def _cap_per_class(detections: list[dict]) -> list[dict]:
    """Keep only top N detections per class by confidence."""
    by_label: dict[str, list[dict]] = {}
    for d in detections:
        lbl = d["label"]
        by_label.setdefault(lbl, []).append(d)

    result: list[dict] = []
    for lbl, dets in by_label.items():
        cap = _MAX_PER_CLASS.get(lbl, 4)  # default 4 for unlisted
        sorted_dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)
        result.extend(sorted_dets[:cap])
    return result


_MIN_AREA_BY_LABEL: dict[str, int] = {
    "door": 400,
    "person": 500,
    "building": 300,
    "roof": 300,
    "house": 300,
}


def _min_area(bbox: dict, label: str = "", min_pixels: int = 1500) -> bool:
    w = bbox["xmax"] - bbox["xmin"]
    h = bbox["ymax"] - bbox["ymin"]
    threshold = _MIN_AREA_BY_LABEL.get(label, min_pixels)
    return w * h >= threshold


def _clip_polygon_to_bounds(pts: list[list[float]], img_w: int, img_h: int) -> list[list[float]] | None:
    """Clip polygon points to image bounds so outlines stay inside the frame."""
    if not pts:
        return None
    clipped = []
    for x, y in pts:
        cx = max(0.0, min(float(img_w), x))
        cy = max(0.0, min(float(img_h), y))
        clipped.append([cx, cy])
    # Remove consecutive duplicates
    deduped = [clipped[0]]
    for p in clipped[1:]:
        if abs(p[0] - deduped[-1][0]) > 1e-6 or abs(p[1] - deduped[-1][1]) > 1e-6:
            deduped.append(p)
    if len(deduped) < 3:
        return None
    return deduped


def _prepare_mask(mask, img_w: int, img_h: int):
    """Shared mask preprocessing: squeeze, threshold, crop, upsample, morphology."""
    if mask is None:
        return None, 0, 0
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        return None, 0, 0
    if arr.dtype != np.uint8:
        arr = (arr > 0.5).astype(np.uint8)
    h_lim, w_lim = min(arr.shape[0], img_h), min(arr.shape[1], img_w)
    arr = arr[:h_lim, :w_lim]
    mh, mw = arr.shape[0], arr.shape[1]
    if mw < img_w or mh < img_h:
        arr = cv2.resize(arr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        arr = (arr > 0.5).astype(np.uint8)
        mh, mw = arr.shape[0], arr.shape[1]
    kernel = np.ones((3, 3), np.uint8)
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    arr = cv2.dilate(arr, kernel)
    return arr, mh, mw


def _contour_to_polygon(cnt, mw: int, mh: int, img_w: int, img_h: int) -> list[list[float]] | None:
    """Convert a single OpenCV contour to a clipped polygon."""
    if len(cnt) < 3:
        return None
    peri = cv2.arcLength(cnt, True)
    epsilon = max(0.5, peri * 0.0005)
    cnt = cv2.approxPolyDP(cnt, epsilon, True)
    if len(cnt) < 3:
        return None
    pts = cnt.reshape(-1, 2).tolist()
    pts = [[float(x), float(y)] for x, y in pts]
    sx = img_w / max(1, mw)
    sy = img_h / max(1, mh)
    pts = [[x * sx, y * sy] for x, y in pts]
    return _clip_polygon_to_bounds(pts, img_w, img_h)


def _mask_to_polygon(mask, img_w: int, img_h: int) -> list[list[float]] | None:
    """Extract the single largest polygon contour from binary mask (street view mode)."""
    arr, mh, mw = _prepare_mask(mask, img_w, img_h)
    if arr is None:
        return None
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    return _contour_to_polygon(cnt, mw, mh, img_w, img_h)


_SAT_MIN_CONTOUR_AREA = 100  # minimum contour area in pixels for satellite buildings


def _mask_to_all_polygons(mask, img_w: int, img_h: int) -> list[dict]:
    """Extract ALL contours from a mask as separate polygons (satellite mode).
    Returns list of {polygon, bbox} dicts — one per detected building."""
    arr, mh, mw = _prepare_mask(mask, img_w, img_h)
    if arr is None:
        return []
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    sx = img_w / max(1, mw)
    sy = img_h / max(1, mh)
    results = []
    for cnt in contours:
        if cv2.contourArea(cnt) < _SAT_MIN_CONTOUR_AREA:
            continue
        poly = _contour_to_polygon(cnt, mw, mh, img_w, img_h)
        if not poly:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        bbox = {
            "xmin": x * sx,
            "ymin": y * sy,
            "xmax": (x + cw) * sx,
            "ymax": (y + ch) * sy,
        }
        results.append({"polygon": poly, "bbox": bbox})
    return results


def run_detection(image_bytes: bytes, mode: str = "streetview") -> dict:
    """
    Run SAM 3 detection on image. Returns dict compatible with DetectionResult:
    { image_width, image_height, detections, processing_time_ms }

    mode: "streetview" for door detection, "satellite" for building detection
    """
    if not load_sam3():
        raise RuntimeError("SAM 3 model not loaded")

    import torch

    prompts = SATELLITE_PROMPTS if mode == "satellite" else STREETVIEW_PROMPTS

    start = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    # Downscale large images for faster inference
    infer_image = image
    scale_x, scale_y = 1.0, 1.0
    if max(w, h) > _MAX_INFER_DIM:
        ratio = _MAX_INFER_DIM / max(w, h)
        infer_w, infer_h = int(w * ratio), int(h * ratio)
        infer_image = image.resize((infer_w, infer_h), Image.Resampling.LANCZOS)
        scale_x = w / infer_w
        scale_y = h / infer_h
    else:
        infer_w, infer_h = w, h

    all_dets: list[dict] = []
    confidence_threshold = 0.3 if mode == "satellite" else 0.55

    for batch_start in range(0, len(prompts), _BATCH_SIZE):
        batch_prompts = prompts[batch_start : batch_start + _BATCH_SIZE]
        batch_images = [infer_image] * len(batch_prompts)

        try:
            inputs = _processor(
                images=batch_images,
                text=batch_prompts,
                return_tensors="pt",
            ).to(_device)

            target_sizes = inputs.get("original_sizes")
            if target_sizes is not None and hasattr(target_sizes, "tolist"):
                target_sizes = target_sizes.tolist()
            else:
                target_sizes = [[infer_h, infer_w]] * len(batch_prompts)

            with torch.inference_mode():
                outputs = _model(**inputs)

            results = _processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence_threshold,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )

            for prompt, result in zip(batch_prompts, results):
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                masks = result.get("masks", [])

                for i, (box, score) in enumerate(zip(boxes, scores)):
                    if score < confidence_threshold:
                        continue

                    if mode == "satellite" and i < len(masks):
                        mask_arr = masks[i]
                        if hasattr(mask_arr, "cpu"):
                            mask_arr = mask_arr.cpu().numpy()
                        sub_polys = _mask_to_all_polygons(mask_arr, infer_w, infer_h)
                        for sp in sub_polys:
                            sb = sp["bbox"]
                            if scale_x != 1.0 or scale_y != 1.0:
                                sb = {
                                    "xmin": sb["xmin"] * scale_x,
                                    "ymin": sb["ymin"] * scale_y,
                                    "xmax": sb["xmax"] * scale_x,
                                    "ymax": sb["ymax"] * scale_y,
                                }
                                sp["polygon"] = [
                                    [px * scale_x, py * scale_y]
                                    for px, py in sp["polygon"]
                                ]
                            all_dets.append({
                                "label": prompt,
                                "confidence": float(score),
                                "bbox": sb,
                                "polygon": sp["polygon"],
                            })
                        continue

                    x1, y1, x2, y2 = box.tolist()
                    bbox = {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}
                    if not _min_area(bbox, label=prompt):
                        continue
                    polygon = None
                    if i < len(masks):
                        mask_arr = masks[i]
                        if hasattr(mask_arr, "cpu"):
                            mask_arr = mask_arr.cpu().numpy()
                        polygon = _mask_to_polygon(mask_arr, infer_w, infer_h)
                    if scale_x != 1.0 or scale_y != 1.0:
                        bbox = {
                            "xmin": bbox["xmin"] * scale_x,
                            "ymin": bbox["ymin"] * scale_y,
                            "xmax": bbox["xmax"] * scale_x,
                            "ymax": bbox["ymax"] * scale_y,
                        }
                        if polygon:
                            polygon = [[px * scale_x, py * scale_y] for px, py in polygon]
                    all_dets.append({
                        "label": prompt,
                        "confidence": float(score),
                        "bbox": bbox,
                        "polygon": polygon,
                    })
        except Exception as e:
            logger.debug(f"Batch {batch_prompts} failed: {e}")
            continue

    if mode == "satellite":
        for d in all_dets:
            if d["label"] in ("roof", "house"):
                d["label"] = "building"
        # Lenient NMS — only remove near-identical duplicates (IoU > 0.7)
        all_dets = _nms(all_dets, iou_threshold=0.7)
    else:
        all_dets = _nms(all_dets, iou_threshold=0.6)
        all_dets = _filter_person_building_overlap(all_dets, w, h)
        all_dets = _filter_google_map_signs(all_dets)
        all_dets = _filter_sign_pole_on_building(all_dets)
        all_dets = _filter_car_doors(all_dets)
        all_dets = _merge_sidewalk_detections(all_dets, h)
    all_dets = _cap_per_class(all_dets)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    detections = []
    for i, d in enumerate(all_dets):
        det = {
            "id": f"det_{i}",
            "label": d["label"],
            "confidence": d["confidence"],
            "bbox": d["bbox"],
        }
        if d.get("polygon"):
            det["polygon"] = d["polygon"]
        detections.append(det)

    return {
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "processing_time_ms": elapsed_ms,
    }
