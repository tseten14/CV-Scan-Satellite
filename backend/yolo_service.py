# Self-trained YOLOv9-Tiny door detector (yolov9t.pt).
# Street view: tuned conf / imgsz + light filters for recall on real entrances.
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from sam3_service import (
    _nms,
    _cap_per_class,
    _env_truthy,
    _filter_first_floor_entrances,
    _filter_non_building_doors,
    _merge_entrance_detections,
)

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


def _filter_streetview_door_false_positives_relaxed(
    dets: list[dict], img_w: int, img_h: int
) -> list[dict]:
    """
    Lighter than strict: drops obvious vehicles / huge slabs / speck noise.
    Keeps more porch / mid-frame doors that strict mode often removed.
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
        cy = 0.5 * (b["ymin"] + b["ymax"])
        rh = bh / max(img_h, 1)

        if ar > 1.25 and rh < 0.13:
            continue
        if ar > 1.0 and bw > 0.44 * img_w:
            continue
        if area_frac > 0.17 and ar > 0.92:
            continue
        if ar > 1.65 and area_frac < 0.055:
            continue
        if ar < 0.13 and area_frac < 0.002:
            continue
        if cy > 0.88 * img_h and d.get("confidence", 0) < 0.32:
            continue
        out.append(d)
    return out


def _apply_yolo_street_shape_filter(dets: list[dict], img_w: int, img_h: int) -> list[dict]:
    mode_f = (os.environ.get("YOLO_STREET_FILTER") or "relaxed").strip().lower()
    if mode_f in ("off", "none", "0", "false", "no"):
        return dets
    if mode_f == "strict":
        return _filter_streetview_door_false_positives(dets, img_w, img_h)
    if mode_f not in ("relaxed", "loose", ""):
        logger.warning("Unknown YOLO_STREET_FILTER=%r, using relaxed", mode_f)
    return _filter_streetview_door_false_positives_relaxed(dets, img_w, img_h)


def _normalize_entrance_labels(dets: list[dict]) -> None:
    for d in dets:
        if d.get("label") == "door":
            d["label"] = "entrance"


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
        # Low default conf: self-trained tiny YOLO often scores <0.2 on real street facades.
        conf = float((os.environ.get("YOLO_STREET_CONF") or "0.10").strip() or 0.10)
        if long_side < 480 or (w * h) < 280_000:
            conf = min(conf, float((os.environ.get("YOLO_STREET_CONF_SMALL") or "0.10").strip() or 0.10))
        iou_nms = float((os.environ.get("YOLO_STREET_IOU") or "0.45").strip() or 0.45)
    else:
        imgsz = int(max(640, min(1280, long_side)))
        imgsz = max(32, (imgsz // 32) * 32)
        conf = float((os.environ.get("YOLO_SAT_CONF") or "0.12").strip() or 0.12)
        iou_nms = 0.55

    conf = max(0.03, min(0.55, conf))
    max_det = 80 if mode == "streetview" else 300
    use_tta = _env_truthy("YOLO_TTA")

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

    def _predict(imgsz_run: int, conf_run: float, iou_run: float) -> list[dict]:
        pred = _model.predict(
            image,
            imgsz=imgsz_run,
            conf=conf_run,
            iou=iou_run,
            max_det=max_det,
            verbose=False,
            augment=use_tta,
        )
        return _extract_dets(pred[0])

    dets = _predict(imgsz, conf, iou_nms)

    # Extra pass at larger square size — small doors benefit (costs another forward).
    if mode == "streetview":
        ms = (os.environ.get("YOLO_MULTISCALE") or "auto").strip().lower()
        extra_sz = min(1280, imgsz + 192)
        extra_sz = max(32, (extra_sz // 32) * 32)
        want_ms = ms in ("1", "true", "yes", "on", "always")
        if not want_ms and ms in ("auto", ""):
            want_ms = len(dets) < 2
        if want_ms and extra_sz >= imgsz + 32:
            conf2 = max(0.035, min(conf * 0.85, conf - 0.03))
            d2 = _predict(extra_sz, conf2, min(0.48, iou_nms + 0.04))
            dets = dets + d2

    # When the model is silent, retry at lower conf — must run for large tiles too (was gated on long_side<640).
    if mode == "streetview" and not dets:
        salv_sz = min(1280, max(imgsz, (int(long_side * 1.12 + 31) // 32) * 32))
        salv_iou = min(0.55, iou_nms + 0.1)
        for c in (
            max(0.025, conf * 0.55),
            0.08,
            0.06,
            0.045,
            0.03,
            0.025,
        ):
            c = max(0.02, min(0.5, float(c)))
            trial = _predict(salv_sz, c, salv_iou)
            if trial:
                logger.info(
                    "YOLO streetview: salvage got %d boxes (conf=%.3f imgsz=%d)",
                    len(trial),
                    c,
                    salv_sz,
                )
                dets = trial
                break

    raw_count = len(dets)
    if mode == "streetview":
        # Single-class door model: treat any detection as a door candidate.
        for d in dets:
            d["label"] = "entrance"
        min_kept_conf = float((os.environ.get("YOLO_STREET_MIN_CONF") or "0.06").strip() or 0.06)
        dets = [d for d in dets if float(d.get("confidence", 0)) >= min_kept_conf]
        dets = _apply_yolo_street_shape_filter(dets, w, h)
        # Align with SAM3 street post-process: drop car-door-shaped boxes, merge double doors, first floor.
        dets = _filter_non_building_doors(dets, w, h)
        dets = _merge_entrance_detections(dets)
        _normalize_entrance_labels(dets)
        try:
            ff_y = float((os.environ.get("YOLO_FIRST_FLOOR_MIN_Y_RATIO") or "0.22").strip())
        except ValueError:
            ff_y = 0.22
        dets = _filter_first_floor_entrances(dets, h, min_center_y_ratio=ff_y)
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

    nms_iou = (
        float((os.environ.get("YOLO_STREET_NMS_IOU") or "0.40").strip() or 0.40)
        if mode == "streetview"
        else 0.55
    )
    nms_iou = max(0.25, min(0.75, nms_iou))
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
    filt_mode = (os.environ.get("YOLO_STREET_FILTER") or "relaxed").strip() or "relaxed"
    logger.info(
        "YOLO: %d objects in %.3fs (%s, %dx%d imgsz=%d conf=%.3f raw_boxes=%d tta=%s filter=%s)",
        len(detections),
        elapsed_s,
        mode,
        w,
        h,
        imgsz,
        conf,
        raw_count,
        use_tta,
        filt_mode,
    )
    if mode == "streetview" and raw_count == 0:
        logger.warning(
            "YOLO streetview: 0 raw boxes from model (weights/domain mismatch or conf still too high; try YOLO_STREET_CONF=0.05 or verify YOLO_WEIGHTS)",
        )
    elif mode == "streetview" and len(detections) == 0:
        logger.warning(
            "YOLO streetview: all %d raw boxes removed by filters (try YOLO_STREET_FILTER=off or lower YOLO_STREET_MIN_CONF)",
            raw_count,
        )

    return {
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "processing_time_s": elapsed_s,
        "engine": "yolo",
    }
