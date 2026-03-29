# Self-trained YOLOv9-Tiny door detector (yolov9t.pt).
# Street view: tuned conf / imgsz + light filters for recall on real entrances.
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from sam3_service import _nms, _cap_per_class

logger = logging.getLogger("uvicorn.error")

_backend_dir = Path(__file__).resolve().parent

# Default weights: prefer yolo-selftrain/ (where training artifacts live), else backend root.
_YOLO_WEIGHT_CANDIDATES = (
    _backend_dir / "yolo-selftrain" / "yolov9t.pt",
    _backend_dir / "yolov9t.pt",
)

_model: Any | None = None


def _resolve_yolo_weights_path() -> Path | None:
    env_path = (os.environ.get("YOLO_WEIGHTS") or "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if not p.is_file():
            logger.error("YOLO_WEIGHTS is not a file: %s", p)
            return None
        return p
    for p in _YOLO_WEIGHT_CANDIDATES:
        if p.is_file():
            return p
    return None


def load_yolo() -> bool:
    """Load YOLOv9 door weights once. Returns True on success."""
    global _model
    if _model is not None:
        return True

    path = _resolve_yolo_weights_path()
    if path is None:
        logger.error(
            "YOLO weights not found. Add yolo-selftrain/yolov9t.pt or yolov9t.pt under backend/, "
            "or set YOLO_WEIGHTS to a .pt file.",
        )
        return False

    try:
        from ultralytics import YOLO
        logger.info("Loading YOLO door model: %s …", path)
        _model = YOLO(str(path))
        logger.info("YOLO ready (%s).", path.name)
        return True
    except Exception as e:
        logger.exception("Failed to load YOLO: %s", e)
        return False


def _bbox_to_polygon(bbox: dict) -> list[list[float]]:
    x0, y0 = bbox["xmin"], bbox["ymin"]
    x1, y1 = bbox["xmax"], bbox["ymax"]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _normalize_label(raw: str) -> str:
    return (raw or "").strip().lower()


def _filter_streetview_door_false_positives(
    dets: list[dict], img_w: int, img_h: int
) -> list[dict]:
    """
    Single-class YOLO fires on people, cars, windows. Keep boxes that look like facade doors:
    portrait-ish, mid-frame (porch), not huge landscape blobs (vehicles).
    """
    img_area = max(img_w * img_h, 1)
    out: list[dict] = []
    for d in dets:
        b = d["bbox"]
        bw = max(0.0, b["xmax"] - b["xmin"])
        bh = max(0.0, b["ymax"] - b["ymin"])
        if bh < 1e-6 or bw < 1e-6:
            continue
        ar = bw / bh  # width / height
        tall = bh / bw
        area_frac = (bw * bh) / img_area
        cx = 0.5 * (b["xmin"] + b["xmax"])
        cy = 0.5 * (b["ymin"] + b["ymax"])
        rh = bh / max(img_h, 1)
        rw = bw / max(img_w, 1)

        # Wide & short → car side / hood / road, not a door.
        if ar > 1.2 and rh < 0.14:
            continue
        if ar > 1.0 and bw > 0.38 * img_w:
            continue
        if area_frac > 0.14 and ar > 0.95:
            continue

        # Tall narrow in lower frame → pedestrians / poles (not porch doors).
        if cy > 0.58 * img_h and tall > 1.85 and rw < 0.22:
            continue
        if cy > 0.62 * img_h and tall > 2.15 and area_frac < 0.055:
            continue

        # Deep foreground strip with modest height → often not building entrance.
        if cy > 0.74 * img_h and tall < 1.35 and ar > 0.85:
            continue

        # Facade doors usually sit above the lower sidewalk band; keep strong boxes low anyway.
        if cy > 0.82 * img_h and d.get("confidence", 0) < 0.42:
            continue

        # Very flat landscape sliver.
        if ar > 1.55 and area_frac < 0.06:
            continue
        # Ultra-narrow vertical noise.
        if ar < 0.16 and area_frac < 0.0025:
            continue

        # Typical door: at least modest height on the house (unless very confident).
        if rh < 0.028 and area_frac < 0.0018 and d.get("confidence", 0) < 0.45:
            continue

        out.append(d)
    return out


def run_yolo_detection(image_bytes: bytes, mode: str = "streetview") -> dict:
    """
    Run self-trained YOLO on image bytes. JSON shape matches SAM run_detection().
    engine is always \"yolo\" for the client.
    """
    if not load_yolo():
        raise RuntimeError(
            "YOLO not loaded. Add yolo-selftrain/yolov9t.pt or yolov9t.pt under backend/, "
            "or set YOLO_WEIGHTS to a .pt file and restart."
        )

    assert _model is not None
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    start = time.perf_counter()
    long_side = max(w, h, 1)
    # Google Street View thumbnails are often 640×640 or smaller; upscaled imgsz helps recall.
    if mode == "streetview":
        if long_side < 560:
            imgsz = int(max(640, min(1280, long_side * 2.25)))
        else:
            imgsz = int(max(640, min(1280, long_side)))
        imgsz = max(32, (imgsz // 32) * 32)
        # Higher default — very low conf floods Street View with people/cars as "door".
        conf = float((os.environ.get("YOLO_STREET_CONF") or "0.22").strip() or 0.22)
        if long_side < 480 or (w * h) < 280_000:
            conf = min(conf, float((os.environ.get("YOLO_STREET_CONF_SMALL") or "0.16").strip() or 0.16))
        iou_nms = float((os.environ.get("YOLO_STREET_IOU") or "0.45").strip() or 0.45)
    else:
        imgsz = int(max(640, min(1280, long_side)))
        imgsz = max(32, (imgsz // 32) * 32)
        conf = float((os.environ.get("YOLO_SAT_CONF") or "0.12").strip() or 0.12)
        iou_nms = 0.55

    conf = max(0.03, min(0.55, conf))
    max_det = 80 if mode == "streetview" else 300

    results = _model.predict(
        image,
        imgsz=imgsz,
        conf=conf,
        iou=iou_nms,
        max_det=max_det,
        verbose=False,
        augment=False,
    )

    def _extract_dets(res) -> list[dict]:
        out: list[dict] = []
        nm = res.names if res.names is not None else {}
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf_sc = float(boxes.conf[i].cpu().numpy())
            cls_i = int(boxes.cls[i].cpu().numpy())
            if isinstance(nm, dict):
                raw_lbl = nm.get(cls_i, str(cls_i))
            else:
                raw_lbl = nm[cls_i] if cls_i < len(nm) else str(cls_i)
            lbl = _normalize_label(str(raw_lbl))
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            out.append({"label": lbl, "confidence": conf_sc, "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}})
        return out

    r = results[0]
    dets = _extract_dets(r)

    # Second pass only for empty small tiles; keep floor well above junk threshold.
    if mode == "streetview" and not dets and long_side < 520:
        retry_conf = max(0.10, min(conf * 0.55, 0.14))
        if retry_conf + 1e-6 < conf:
            logger.info("YOLO streetview: no boxes at conf=%.3f, retry conf=%.3f", conf, retry_conf)
            r2 = _model.predict(
                image,
                imgsz=imgsz,
                conf=retry_conf,
                iou=min(0.50, iou_nms + 0.05),
                max_det=max_det,
                verbose=False,
                augment=False,
            )[0]
            dets = _extract_dets(r2)

    raw_count = len(dets)
    if mode == "streetview":
        # Single-class door model: treat any detection as a door candidate.
        for d in dets:
            d["label"] = "entrance"
        min_kept_conf = float((os.environ.get("YOLO_STREET_MIN_CONF") or "0.14").strip() or 0.14)
        dets = [d for d in dets if float(d.get("confidence", 0)) >= min_kept_conf]
        dets = _filter_streetview_door_false_positives(dets, w, h)
    elif mode == "satellite":
        for d in dets:
            d["label"] = "building"
        img_area = w * h
        dets = [
            d
            for d in dets
            if (d["bbox"]["xmax"] - d["bbox"]["xmin"]) * (d["bbox"]["ymax"] - d["bbox"]["ymin"])
            < 0.12 * img_area
        ]

    nms_iou = 0.42 if mode == "streetview" else 0.55
    dets = _nms(dets, iou_threshold=nms_iou)
    dets = _cap_per_class(dets)
    if mode == "streetview" and len(dets) > 8:
        dets = sorted(dets, key=lambda x: float(x.get("confidence", 0)), reverse=True)[:8]

    detections: list[dict] = []
    for i, d in enumerate(dets):
        detections.append({
            "id": f"det_{i}",
            "label": d["label"],
            "confidence": d["confidence"],
            "bbox": d["bbox"],
            "polygon": _bbox_to_polygon(d["bbox"]),
        })

    elapsed_s = round(time.perf_counter() - start, 3)
    logger.info(
        "YOLO: %d objects in %.3fs (%s, %dx%d imgsz=%d conf=%.3f raw_boxes=%d)",
        len(detections),
        elapsed_s,
        mode,
        w,
        h,
        imgsz,
        conf,
        raw_count,
    )

    return {
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "processing_time_s": elapsed_s,
        "engine": "yolo",
    }
