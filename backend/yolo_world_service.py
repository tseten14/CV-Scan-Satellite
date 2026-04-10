# YOLO v8-world: open-vocabulary detection for doors / entrances without custom .pt weights.
# Default hub weights: yolov8s-worldv2.pt (Ultralytics download on first use).
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from sam3_service import _nms, _cap_per_class
from yolo_service import _bbox_to_polygon, _normalize_label

logger = logging.getLogger("uvicorn.error")

_backend_dir = Path(__file__).resolve().parent

_model: Any | None = None
_model_path_resolved: str = ""


def _world_model_ctor() -> Callable[..., Any]:
    try:
        from ultralytics import YOLOWorld

        return YOLOWorld  # type: ignore[return-value]
    except ImportError:
        from ultralytics import YOLO

        return YOLO


def _resolve_world_weights() -> str:
    env_path = (os.environ.get("YOLO_WORLD_WEIGHTS") or "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if not p.is_file():
            logger.error("YOLO_WORLD_WEIGHTS is not a file: %s", p)
            return ""
        return str(p)
    # Hub id or local filename — Ultralytics downloads known names on first use.
    return (os.environ.get("YOLO_WORLD_MODEL") or "yolov8s-worldv2.pt").strip() or "yolov8s-worldv2.pt"


def _parse_class_list(raw: str) -> list[str]:
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


def _street_class_prompts() -> list[str]:
    default = (
        "door,front door,house door,building entrance,porch door,entryway,doorway,"
        "glass door,storefront,revolving door,entrance"
    )
    return _parse_class_list(os.environ.get("YOLO_WORLD_STREET_CLASSES") or default)


def _light_world_street_filter(dets: list[dict], img_w: int, img_h: int) -> list[dict]:
    """
    YOLO v9 uses tight façade heuristics that assume single-class door scores (~0.2–0.9).
    YOLO v8-world scores are usually much lower; the v9 filter would drop almost everything.
    Here we only remove obvious huge / landscape junk.
    """
    img_area = max(img_w * img_h, 1)
    out: list[dict] = []
    for d in dets:
        b = d["bbox"]
        bw = max(0.0, b["xmax"] - b["xmin"])
        bh = max(0.0, b["ymax"] - b["ymin"])
        if bh < 1e-6 or bw < 1e-6:
            continue
        ar = bw / bh
        area_frac = (bw * bh) / img_area
        rh = bh / max(img_h, 1)
        # Full-frame or near-full blobs (sky + facade mistaken as one box).
        if area_frac > 0.42 and ar > 0.75:
            continue
        # Very flat road/hood-like slivers.
        if ar > 2.4 and rh < 0.07:
            continue
        out.append(d)
    return out


def _satellite_class_prompts() -> list[str]:
    default = "building,roof,warehouse,structure"
    return _parse_class_list(os.environ.get("YOLO_WORLD_SAT_CLASSES") or default)


def load_yolo_world() -> bool:
    global _model, _model_path_resolved
    if _model is not None:
        return True

    path = _resolve_world_weights()
    if not path:
        logger.error(
            "YOLO v8-world weights missing. Set YOLO_WORLD_WEIGHTS to a .pt file, "
            "or YOLO_WORLD_MODEL to a hub name (default yolov8s-worldv2.pt).",
        )
        return False

    try:
        ctor = _world_model_ctor()
        logger.info("Loading YOLO v8-world: %s …", path)
        _model = ctor(path)
        _model_path_resolved = path
        logger.info("YOLO v8-world ready (%s).", Path(path).name)
        return True
    except Exception as e:
        logger.exception("Failed to load YOLO v8-world: %s", e)
        _model = None
        _model_path_resolved = ""
        return False


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


def run_yolo_world_detection(image_bytes: bytes, mode: str = "streetview") -> dict:
    """
    Open-vocabulary YOLO v8-world. JSON shape matches SAM / YOLO v9 door pipeline.
    engine is \"yolo_world\" for the client.
    """
    if not load_yolo_world():
        raise RuntimeError(
            "YOLO v8-world not loaded. Set YOLO_WORLD_WEIGHTS to a local .pt file, "
            "or install ultralytics and use default hub weights (yolov8s-worldv2.pt). "
            "First run may download weights."
        )

    assert _model is not None
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    classes = _street_class_prompts() if mode == "streetview" else _satellite_class_prompts()
    if not classes:
        raise ValueError("YOLO v8-world: no class prompts configured for this mode.")

    try:
        _model.set_classes(classes)
    except Exception as e:
        logger.exception("YOLO v8-world set_classes failed: %s", e)
        raise RuntimeError(
            "YOLO v8-world set_classes failed — check ultralytics version and YOLO v8-world support."
        ) from e

    start = time.perf_counter()
    long_side = max(w, h, 1)
    imgsz = int(max(640, min(1280, long_side)))
    imgsz = max(32, (imgsz // 32) * 32)

    if mode == "streetview":
        # v8-world confidences are typically lower than a fine-tuned v9 door head — default lower floor.
        conf = float((os.environ.get("YOLO_WORLD_STREET_CONF") or "0.06").strip() or 0.06)
        iou_nms = float((os.environ.get("YOLO_WORLD_STREET_IOU") or "0.45").strip() or 0.45)
    else:
        conf = float((os.environ.get("YOLO_WORLD_SAT_CONF") or "0.10").strip() or 0.10)
        iou_nms = float((os.environ.get("YOLO_WORLD_SAT_IOU") or "0.55").strip() or 0.55)

    conf = max(0.03, min(0.55, conf))
    max_det = 80 if mode == "streetview" else 300

    def _predict(conf_v: float):
        return _model.predict(
            image,
            imgsz=imgsz,
            conf=conf_v,
            iou=iou_nms,
            max_det=max_det,
            verbose=False,
            augment=False,
        )

    results = _predict(conf)
    r = results[0]
    dets = _extract_dets(r)

    # Second pass when the first yields nothing (common on residential façades at default conf).
    if mode == "streetview" and not dets:
        retry_conf = max(0.03, min(conf * 0.5, 0.05))
        if retry_conf + 1e-6 < conf:
            logger.info(
                "YOLO v8-world streetview: no boxes at conf=%.3f, retry conf=%.3f",
                conf,
                retry_conf,
            )
            r = _predict(retry_conf)[0]
            dets = _extract_dets(r)

    raw_count = len(dets)

    if mode == "streetview":
        for d in dets:
            d["label"] = "entrance"
        min_kept = float((os.environ.get("YOLO_WORLD_STREET_MIN_CONF") or "0.03").strip() or 0.03)
        dets = [d for d in dets if float(d.get("confidence", 0)) >= min_kept]
        use_v9_filter = (os.environ.get("YOLO_WORLD_USE_V9_FACADE_FILTER") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if use_v9_filter:
            from yolo_service import _filter_streetview_door_false_positives

            dets = _filter_streetview_door_false_positives(dets, w, h)
        else:
            dets = _light_world_street_filter(dets, w, h)
    else:
        for d in dets:
            d["label"] = "building"
        try:
            max_bbox_frac = float((os.environ.get("YOLO_WORLD_SAT_MAX_BBOX_FRACTION") or "0.62").strip())
        except ValueError:
            max_bbox_frac = 0.62
        max_bbox_frac = max(0.18, min(0.92, max_bbox_frac))
        img_area = w * h
        dets = [
            d
            for d in dets
            if (d["bbox"]["xmax"] - d["bbox"]["xmin"]) * (d["bbox"]["ymax"] - d["bbox"]["ymin"])
            <= max_bbox_frac * img_area
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
        "YOLO v8-world: %d objects in %.3fs (%s, %dx%d imgsz=%d conf=%.3f prompts=%d raw=%d)",
        len(detections),
        elapsed_s,
        mode,
        w,
        h,
        imgsz,
        conf,
        len(classes),
        raw_count,
    )

    return {
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "processing_time_s": elapsed_s,
        "engine": "yolo_world",
    }
