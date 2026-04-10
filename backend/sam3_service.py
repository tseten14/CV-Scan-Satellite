
# SAM 3 (Segment Anything Model 3) detection service.
# Uses Hugging Face transformers for promptable concept segmentation.
import os
import io
import time
import logging
import contextlib
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("uvicorn.error")

# Entrance-focused prompts for street view mode.

# SAM 3 is promptable: for each text prompt, the model returns instance segmentation
# masks for any regions matching the prompt.

# Street-level imagery contains many “door-like”/“entrance-like” visual patterns
# (including reflective glass storefronts), so we use multiple entrance synonyms.
# Do NOT mix “car” prompts here — that would label vehicle panels as the same task as façades.
STREETVIEW_PROMPTS = [
    "door",
    "building entrance",
]

# Optional second pass (off by default for speed). Minimal prompts when enabled.
STREETVIEW_VEHICLE_PROMPTS = [
    "car",
    "vehicle",
]

# Broad prompts improve recall on varied roof types (residential, warehouse, retail).
# All labels are merged to "building" before return.
SATELLITE_PROMPTS = [
    "building",
    "roof",
    "warehouse",
    "structure",
]

_BATCH_SIZE = 2
# Default 384: faster CPU runs (~<20s target on typical tiles); raise SAM3_STREET_MAX_DIM for quality.
_DEFAULT_STREET_MAX_DIM = 384
_DEFAULT_VEHICLE_INFER_MAX_DIM = 288
# Legacy alias used in comments / env docs
_MAX_INFER_DIM = _DEFAULT_STREET_MAX_DIM

_model: Any = None
_processor: Any = None
_model_id: str = ""

# Device selection impacts both speed and memory usage.
_device: str = "cpu"
_dtype: Any = None

# SAM 3.1 has no Transformers integration yet (raw checkpoints only on HF).
# Using facebook/sam3 which is the latest Transformers-compatible release.
SAM3_MODEL_ID = "facebook/sam3"


def _get_device() -> str:
    # Default CPU (no Apple MPS) — cooler, avoids GPU thermal spikes. Override if needed:
    # SAM3_DEVICE=cuda | mps | auto  (auto = CUDA then MPS then CPU)
    raw = (os.environ.get("SAM3_DEVICE") or "cpu").strip().lower()
    try:
        import torch
        if raw == "cpu":
            return "cpu"
        if raw == "cuda" and torch.cuda.is_available():
            return "cuda"
        if raw == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if raw == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if raw in ("cuda", "mps"):
            logger.warning("SAM3_DEVICE=%s not available; using CPU", raw)
        return "cpu"
    except Exception:
        return "cpu"


def load_sam3() -> bool:
    # Load SAM 3 model and processor. Returns True on success.
    global _model, _processor, _device, _dtype, _model_id
    if _model is not None:
        return True

    try:
        import torch
        from transformers import Sam3Model, Sam3Processor

        _device = _get_device()
        _apply_runtime_footprint(_device)
        # float16 on MPS triggers torch.arange precision bugs in SAM3's decoder;
        # restrict to CUDA where it is well-tested.
        _dtype = torch.float16 if _device == "cuda" else torch.float32
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token

        logger.info("Loading SAM model %s on %s (%s)…", SAM3_MODEL_ID, _device, _dtype)
        _processor = Sam3Processor.from_pretrained(
            SAM3_MODEL_ID,
            token=token,
        )
        _model = Sam3Model.from_pretrained(
            SAM3_MODEL_ID,
            token=token,
            torch_dtype=_dtype,
        ).to(_device)
        _model.eval()
        if _device == "cuda":
            torch.backends.cudnn.benchmark = True
        _model_id = SAM3_MODEL_ID
        if _device == "cuda" and (os.environ.get("SAM3_TORCH_COMPILE") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                _model = torch.compile(_model, mode="reduce-overhead")  # type: ignore[assignment]
                logger.info("SAM model torch.compile (CUDA) enabled")
            except Exception as e:
                logger.warning("SAM3 torch.compile skipped: %s", e)
        logger.info("SAM model ready: %s", _model_id)
        return True
    except Exception as e:
        logger.error(f"SAM 3 failed to load: {e}")
        return False


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes")


def _street_vehicle_pass_enabled() -> bool:
    # Off by default — halves SAM3 work on street view (big win for CPU latency & thermals).
    # SAM3_VEHICLE_PASS=1 enables car/vehicle mask pass for door-on-car filtering.
    if _env_truthy("SAM3_VEHICLE_PASS") or _env_truthy("SAM3_RUN_VEHICLE_PASS"):
        return True
    if _env_truthy("SAM3_SKIP_VEHICLE_PASS"):
        return False
    legacy = (os.environ.get("SAM3_SKIP_VEHICLE_PASS") or "").strip().lower()
    if legacy in ("0", "false", "no", "off"):
        return True
    return False


def _system_friendly_default(device: str) -> bool:
    # Cap BLAS/thread usage so the laptop stays usable and CPU package power stays lower.
    raw = (os.environ.get("SAM3_SYSTEM_FRIENDLY") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return True


def _apply_runtime_footprint(device: str) -> None:
    import torch

    friendly = _system_friendly_default(device)
    intra_raw = (os.environ.get("SAM3_INTRA_THREADS") or "").strip()
    n: int | None = None
    if intra_raw:
        try:
            n = max(1, int(intra_raw))
        except ValueError:
            n = None
    elif friendly:
        cpus = os.cpu_count() or 4
        # CPU inference: prefer fewer physical threads to reduce heat; MPS can use a bit more.
        if device == "cpu":
            n = max(2, min(4, max(1, cpus // 3)))
        else:
            n = max(2, min(4, max(1, cpus // 2)))
    if n is not None:
        try:
            torch.set_num_threads(n)
        except Exception:
            pass
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ.setdefault(var, str(n))
    inter_raw = (os.environ.get("SAM3_INTEROP_THREADS") or "").strip()
    if inter_raw:
        try:
            torch.set_num_interop_threads(max(1, int(inter_raw)))
        except (ValueError, RuntimeError):
            pass
    elif friendly:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if friendly:
        logger.info(
            "SAM3 system-friendly CPU limits active (SAM3_SYSTEM_FRIENDLY=0 for max CPU parallelism)"
        )


def _release_accelerator_memory() -> None:
    try:
        import torch

        if _device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif _device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


@contextlib.contextmanager
def _sam3_inference_scope():
    try:
        yield
    finally:
        _release_accelerator_memory()


def _iou(box_a: dict, box_b: dict) -> float:
    # Compute IoU between two boxes (xmin, ymin, xmax, ymax).
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


def _nms(
    detections: list[dict],
    iou_threshold: float = 0.5,
    *,
    entrance_suppress_iou: float | None = None,
) -> list[dict]:
    # Non-maximum suppression (NMS) by confidence.
    
    # Why this is customized:
    # - Urban imagery generates many *overlapping* boxes for the same physical structure.
    # - However, adjacent structures (e.g., two neighboring buildings) may also overlap
    # slightly in the model's bbox space.
    
    # To avoid deleting legitimate neighboring buildings, we use:
    # - a higher threshold for "road"/"sidewalk" to keep continuous surface regions
    # - a more lenient threshold for "building" so neighbors survive
    # - a stricter threshold for other same-class labels
    #
    # entrance_suppress_iou: optional higher IoU bar for two "entrance" boxes (YOLO v8-world:
    # door vs adjacent window on same porch).
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep: list[dict] = []
    for det in sorted_dets:
        keep_it = True
        for k in keep:
            iou = _iou(det["bbox"], k["bbox"])
            thresh = iou_threshold
            if {det["label"], k["label"]} <= {"road", "sidewalk"}:
                thresh = 0.85
            elif det["label"] == k["label"]:
                # Buildings need lenient NMS — adjacent buildings have slight overlap
                if det["label"] == "building":
                    thresh = 0.55
                elif det["label"] == "entrance" and entrance_suppress_iou is not None:
                    thresh = entrance_suppress_iou
                else:
                    thresh = 0.35
            if iou > thresh:
                keep_it = False
                break
        if keep_it:
            keep.append(det)
    return keep


def _overlap_ratio(box_a: dict, box_b: dict) -> float:
    # Intersection over box_a area (how much of A is covered by B).
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
    # Filter out common street-view false positives where the model mistakes part of a facade
    # for a "person" concept.
    
    # We keep "person" detections only if:
    # - their bbox area is not unrealistically large
    # - they are not almost entirely contained inside a detected building bbox
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
    # Remove sign detections that come from Street View navigation UI overlays.
    
    # Those arrow graphics often lie on road/pavement tiles; if a detected "sign" bbox
    # overlaps a road bbox beyond a threshold, we drop it.
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
    # Filter sign/pole detections that are likely false positives caused by overlaps
    # with building edges.
    
    # If a sign/pole bbox overlaps building bboxes too much, we discard it.
    buildings = [d for d in detections if d["label"] == "building"]
    signs = [d for d in detections if d["label"] == "sign"]
    poles = [d for d in detections if d["label"] == "pole"]
    others = [d for d in detections if d["label"] not in ("sign", "pole", "building")]

    def keep_det(d: dict) -> bool:
        return not any(_overlap_ratio(d["bbox"], b["bbox"]) > 0.35 for b in buildings)

    filtered_signs = [s for s in signs if keep_det(s)]
    filtered_poles = [p for p in poles if keep_det(p)]

    return others + buildings + filtered_signs + filtered_poles


def _filter_non_building_doors(detections: list[dict], img_w: int, img_h: int) -> list[dict]:
    # Remove door/entrance detections that are likely vehicle doors or other non-building
    # features based on shape: building doors are portrait-oriented (taller than wide),
    # while car doors/windows are landscape-oriented and small.
    img_area = max(img_w * img_h, 1)
    kept: list[dict] = []
    for d in detections:
        if not _is_entrance_label(d.get("label")):
            kept.append(d)
            continue
        bw = d["bbox"]["xmax"] - d["bbox"]["xmin"]
        bh = d["bbox"]["ymax"] - d["bbox"]["ymin"]
        aspect = bw / max(bh, 1)
        area = bw * bh
        area_frac = area / img_area
        cy = 0.5 * (d["bbox"]["ymin"] + d["bbox"]["ymax"]) / max(img_h, 1)
        if aspect > 1.3 and area < 0.04 * img_area:
            continue
        # Parked-car side panels: modest area, not very portrait, sitting in street band.
        if cy > 0.54 and aspect > 0.42 and aspect < 1.15 and area_frac < 0.055:
            continue
        if cy > 0.58 and aspect < 1.05 and area_frac < 0.04:
            continue
        kept.append(d)
    return kept


_VEHICLE_LABELS = frozenset({
    "car",
    "truck",
    "vehicle",
    "suv",
    "van",
    "automobile",
    "car door",
    "vehicle door",
})


def _expand_bbox_xyxy(
    b: dict, img_w: int, img_h: int, *, rel_w: float = 0.12, rel_h: float = 0.10
) -> dict:
    w = max(1e-6, b["xmax"] - b["xmin"])
    h = max(1e-6, b["ymax"] - b["ymin"])
    cx = 0.5 * (b["xmin"] + b["xmax"])
    cy = 0.5 * (b["ymin"] + b["ymax"])
    nw = w * (1.0 + 2.0 * rel_w)
    nh = h * (1.0 + 2.0 * rel_h)
    return {
        "xmin": max(0.0, cx - nw * 0.5),
        "ymin": max(0.0, cy - nh * 0.5),
        "xmax": min(float(img_w), cx + nw * 0.5),
        "ymax": min(float(img_h), cy + nh * 0.5),
    }


def _point_in_bbox(px: float, py: float, b: dict) -> bool:
    return b["xmin"] <= px <= b["xmax"] and b["ymin"] <= py <= b["ymax"]


def _nms_streetview_split_vehicle(dets: list[dict]) -> list[dict]:
    """NMS entrance prompts vs vehicle prompts separately so IoU does not drop car boxes."""
    veh = [d for d in dets if str(d.get("label", "")).strip().lower() in _VEHICLE_LABELS]
    rest = [d for d in dets if str(d.get("label", "")).strip().lower() not in _VEHICLE_LABELS]
    return _nms(rest, iou_threshold=0.6) + _nms(veh, iou_threshold=0.55)


def _filter_entrances_on_vehicles(detections: list[dict], img_w: int, img_h: int) -> list[dict]:
    """Drop entrance/door masks that sit on detected vehicles (car doors, not building doors)."""
    img_area = max(img_w * img_h, 1)
    vehicles = [
        d for d in detections if str(d.get("label", "")).strip().lower() in _VEHICLE_LABELS
    ]
    if not vehicles:
        return detections

    def vehicle_area_frac(vb: dict) -> float:
        aw = max(0.0, vb["xmax"] - vb["xmin"])
        ah = max(0.0, vb["ymax"] - vb["ymin"])
        return (aw * ah) / img_area

    kept: list[dict] = []
    for d in detections:
        if not _is_entrance_label(d.get("label")):
            kept.append(d)
            continue
        drop = False
        eb = d["bbox"]
        ecx = 0.5 * (eb["xmin"] + eb["xmax"])
        ecy = 0.5 * (eb["ymin"] + eb["ymax"])
        for v in vehicles:
            vb = v["bbox"]
            v_frac = vehicle_area_frac(vb)
            # Big masks that spill onto the façade need a *tight* hull; small masks can grow more.
            if v_frac > 0.24:
                hull = _expand_bbox_xyxy(vb, img_w, img_h, rel_w=0.035, rel_h=0.035)
            else:
                hull = _expand_bbox_xyxy(vb, img_w, img_h, rel_w=0.14, rel_h=0.11)
            if _point_in_bbox(ecx, ecy, hull):
                drop = True
                break
            if _iou(eb, vb) > 0.05:
                drop = True
                break
            if _overlap_ratio(eb, vb) > 0.14:
                drop = True
                break
        if not drop:
            kept.append(d)

    return kept


_ENTRANCE_LABELS = {
    "door",
    "revolving door",
    "glass entrance",
    "storefront entrance",
    "building entrance",
    "entrance",
}


def _is_entrance_label(label: Any) -> bool:
    return str(label or "").strip().lower() in _ENTRANCE_LABELS


def _filter_street_entrance_foreground_clutter(
    detections: list[dict], img_w: int, img_h: int
) -> list[dict]:
    """
    SAM3 often labels car windows / side glass as "door". Those boxes sit low in the frame and
    are often wider-than-tall or only modestly portrait. Building porch doors are usually higher
    (smaller cy) and more portrait than curb-side vehicle glass.
    """
    img_area = max(img_w * img_h, 1)
    kept: list[dict] = []
    for d in detections:
        if not _is_entrance_label(d.get("label")):
            kept.append(d)
            continue
        b = d["bbox"]
        bw = max(1e-6, b["xmax"] - b["xmin"])
        bh = max(1e-6, b["ymax"] - b["ymin"])
        ar = bw / bh
        cy = 0.5 * (b["ymin"] + b["ymax"]) / max(img_h, 1)
        area_frac = (bw * bh) / img_area
        if cy > 0.56 and ar >= 0.82 and area_frac <= 0.06:
            continue
        if cy > 0.60 and ar >= 0.72:
            continue
        if cy > 0.66 and ar >= 0.58:
            continue
        if cy > 0.70 and area_frac < 0.045 and ar > 0.5:
            continue
        y_bottom = b["ymax"] / max(img_h, 1)
        if cy > 0.50 and y_bottom > 0.78 and ar >= 0.40 and area_frac <= 0.07:
            continue
        if 0.52 <= cy <= 0.82 and 0.36 <= ar <= 1.05 and area_frac <= 0.09:
            continue
        kept.append(d)
    return kept


def _merge_entrance_detections(detections: list[dict]) -> list[dict]:
    # Merge multiple overlapping entrance detections into a single entrance.
    
    # This is mainly to avoid separate boxes for each leaf of a glass double-door.
    # We keep the highest-confidence detection and expand its bbox to cover the union.
    entrances = [d for d in detections if _is_entrance_label(d.get("label"))]
    others = [d for d in detections if not _is_entrance_label(d.get("label"))]

    if not entrances:
        return detections

    # Sort by confidence so first seen is the strongest instance
    entrances_sorted = sorted(entrances, key=lambda d: d["confidence"], reverse=True)
    merged: list[dict] = []

    for det in entrances_sorted:
        placed = False
        for m in merged:
            # Use IoU to decide whether this is part of the same physical entrance
            if _iou(det["bbox"], m["bbox"]) > 0.3:
                b = det["bbox"]
                mb = m["bbox"]
                m["bbox"] = {
                    "xmin": min(mb["xmin"], b["xmin"]),
                    "ymin": min(mb["ymin"], b["ymin"]),
                    "xmax": max(mb["xmax"], b["xmax"]),
                    "ymax": max(mb["ymax"], b["ymax"]),
                }
                # Keep polygon of the higher-confidence detection (already m)
                placed = True
                break
        if not placed:
            merged.append(det)

    # Normalize label to a single semantic class for UI consistency
    for m in merged:
        m["label"] = "door"

    return others + merged


def _filter_first_floor_entrances(
    detections: list[dict],
    img_h: int,
    *,
    min_center_y_ratio: float | None = None,
    max_center_y_ratio: float | None = None,
) -> list[dict]:
    # Keep entrances whose center sits in a vertical band: below upper stories / sky, above
    # deep foreground (parked cars). Optional max_center_y_ratio drops curb-hood clutter.
    ratio_lo = 0.32 if min_center_y_ratio is None else max(0.0, min(0.55, float(min_center_y_ratio)))
    ratio_hi: float | None = None
    if max_center_y_ratio is not None:
        ratio_hi = max(0.40, min(0.90, float(max_center_y_ratio)))

    entrances = [d for d in detections if _is_entrance_label(d.get("label"))]
    others = [d for d in detections if not _is_entrance_label(d.get("label"))]

    kept: list[dict] = []
    y_min_keep = ratio_lo * img_h
    for e in entrances:
        b = e["bbox"]
        yc = (b["ymin"] + b["ymax"]) / 2.0
        if yc < y_min_keep:
            continue
        if ratio_hi is not None and yc > ratio_hi * img_h:
            continue
        # Whole box in upper facade → upper-story openings / roofline, not walk-up doors.
        if b["ymax"] <= 0.33 * img_h:
            continue
        kept.append(e)

    return others + kept


def _street_entrance_quality_score(d: dict, img_w: int, img_h: int) -> float:
    """Higher = more likely the main building door (not car glass / neighbor)."""
    b = d["bbox"]
    bw = max(1e-6, b["xmax"] - b["xmin"])
    bh = max(1e-6, b["ymax"] - b["ymin"])
    ar = bw / bh
    area_frac = (bw * bh) / max(img_w * img_h, 1)
    cx = 0.5 * (b["xmin"] + b["xmax"]) / max(img_w, 1)
    cy = 0.5 * (b["ymin"] + b["ymax"]) / max(img_h, 1)
    conf = float(d.get("confidence", 0.0))

    s = conf
    if ar < 0.88:
        s += 0.14
    if ar < 0.68:
        s += 0.06
    if ar > 1.05:
        s -= 0.32
    if ar > 1.28:
        s -= 0.14
    horiz = abs(cx - 0.5)
    s += max(0.0, 0.16 - horiz * 0.45)
    if cx < 0.06 or cx > 0.94:
        s -= 0.20
    if 0.003 <= area_frac <= 0.09:
        s += 0.08
    if area_frac < 0.0009:
        s -= 0.15
    if area_frac > 0.12:
        s -= 0.12
    # Foreground vehicles live in the bottom of the frame — penalize hard.
    if cy > 0.52:
        s -= (cy - 0.52) * 1.1
    if cy > 0.62:
        s -= 0.15
    if cy > 0.72:
        s -= 0.20
    if cy > 0.88:
        s -= 0.25
    if cy < 0.18:
        s -= 0.06
    if 0.34 <= cy <= 0.66:
        s += 0.10
    return s


def _select_single_best_street_entrance(
    detections: list[dict], img_w: int, img_h: int
) -> list[dict]:
    """
    After SAM3 + filters, keep at most one entrance: the most plausible main building door.
    Favors portrait boxes, horizontal center (subject facade), confidence, sane area, and
    vertical position above typical car bands.
    """
    entrances = [d for d in detections if _is_entrance_label(d.get("label"))]
    if not entrances:
        return []

    best = max(entrances, key=lambda d: _street_entrance_quality_score(d, img_w, img_h))
    if len(entrances) > 1:
        logger.info(
            "SAM3 streetview: chose 1 of %d entrance candidates (conf=%.3f)",
            len(entrances),
            float(best.get("confidence", 0.0)),
        )
    return [best]


def _collapse_api_entrances_to_one(
    detections: list[dict], img_w: int, img_h: int
) -> list[dict]:
    """Last line of defense: never return more than one entrance-like detection on street view."""
    ent = [
        d
        for d in detections
        if _is_entrance_label(d.get("label"))
        or str(d.get("label", "")).strip().lower() == "entrance"
    ]
    if len(ent) <= 1:
        out = list(detections)
        for i, d in enumerate(out):
            d["id"] = f"det_{i}"
        return out

    best = max(ent, key=lambda d: _street_entrance_quality_score(d, img_w, img_h))
    ent_ids = {id(x) for x in ent}
    rest = [d for d in detections if id(d) not in ent_ids]
    best_out = {**best, "label": "entrance", "id": "det_0"}
    out = rest + [best_out]
    for i, d in enumerate(out):
        d["id"] = f"det_{i}"
    return out


def _prefer_facade_entrance_cluster(detections: list[dict], img_h: int) -> list[dict]:
    """
    Parked cars sit *lower* in the frame than background house doors. If entrance centers form two
    vertical groups with a clear gap, keep the upper (façade) group only.
    """
    ent = [d for d in detections if _is_entrance_label(d.get("label"))]
    others = [d for d in detections if not _is_entrance_label(d.get("label"))]
    if len(ent) < 2:
        return detections
    cys = [
        (
            d,
            (d["bbox"]["ymin"] + d["bbox"]["ymax"]) / (2.0 * max(img_h, 1)),
        )
        for d in ent
    ]
    cys.sort(key=lambda x: x[1])
    gaps: list[tuple[float, int]] = []
    for i in range(len(cys) - 1):
        gaps.append((cys[i + 1][1] - cys[i][1], i))
    max_gap, idx_at = max(gaps)
    try:
        min_gap = float((os.environ.get("SAM3_FACADE_CY_GAP") or "0.048").strip())
    except ValueError:
        min_gap = 0.048
    min_gap = max(0.03, min(0.14, min_gap))
    if max_gap < min_gap:
        return detections
    upper = [cys[j][0] for j in range(0, idx_at + 1)]
    lower = [cys[j][0] for j in range(idx_at + 1, len(cys))]
    mean_lo = sum(
        (d["bbox"]["ymin"] + d["bbox"]["ymax"]) / (2.0 * max(img_h, 1)) for d in lower
    ) / len(lower)
    try:
        thresh = float((os.environ.get("SAM3_FOREGROUND_CLUSTER_MEAN_CY") or "0.45").strip())
    except ValueError:
        thresh = 0.45
    if mean_lo >= thresh and len(upper) >= 1:
        logger.info(
            "SAM3 streetview: façade cluster kept %d entrance(s), dropped %d lower-band",
            len(upper),
            len(lower),
        )
        return others + upper
    return detections


def _drop_entrances_when_only_foreground_band(
    detections: list[dict], img_h: int
) -> list[dict]:
    """
    If every entrance sits low in the frame (curb / vehicle body) and none sit in the façade band,
    clear them — better no box than only car doors. Single huge FP uses a slightly looser cut.
    """
    ent = [d for d in detections if _is_entrance_label(d.get("label"))]
    if not ent:
        return detections
    cys = [(d["bbox"]["ymin"] + d["bbox"]["ymax"]) / (2.0 * max(img_h, 1)) for d in ent]
    mc = min(cys)
    try:
        thresh_multi = float((os.environ.get("SAM3_MIN_FACADE_CY_ANY") or "0.52").strip())
    except ValueError:
        thresh_multi = 0.52
    try:
        thresh_single = float((os.environ.get("SAM3_SINGLE_ENTRANCE_MAX_CY") or "0.52").strip())
    except ValueError:
        thresh_single = 0.52
    thresh_multi = max(0.32, min(0.55, thresh_multi))
    thresh_single = max(0.42, min(0.62, thresh_single))
    drop = (len(ent) >= 2 and mc > thresh_multi) or (len(ent) == 1 and mc > thresh_single)
    if drop:
        logger.warning(
            "SAM3 streetview: dropped %d entrance(s) — all in foreground band (min cy_norm=%.3f)",
            len(ent),
            mc,
        )
        return [d for d in detections if not _is_entrance_label(d.get("label"))]
    return detections


def _drop_entrances_far_below_best_confidence(
    detections: list[dict], img_h: int
) -> list[dict]:
    """
    Car doors often score almost as high as the real door (e.g. 0.92 vs 0.84 vs 0.80), so a pure
    confidence gap is not enough. Also drop detections that sit clearly *lower* in the frame than
    the best-scoring box and are even slightly weaker — typical curb / vehicle side glass.
    """
    ent = [d for d in detections if _is_entrance_label(d.get("label"))]
    if len(ent) < 2:
        return detections
    best = max(ent, key=lambda d: float(d.get("confidence", 0)))
    best_id = id(best)
    cb = float(best.get("confidence", 0))
    cy_b = (best["bbox"]["ymin"] + best["bbox"]["ymax"]) / (2.0 * max(img_h, 1))
    try:
        gap = float((os.environ.get("SAM3_ENTRANCE_CONF_GAP_DROP") or "0.10").strip())
    except ValueError:
        gap = 0.10
    gap = max(0.05, min(0.40, gap))
    try:
        cy_slop = float((os.environ.get("SAM3_ENTRANCE_CY_BELOW_BEST") or "0.024").strip())
    except ValueError:
        cy_slop = 0.024
    cy_slop = max(0.012, min(0.12, cy_slop))
    try:
        wk = float((os.environ.get("SAM3_ENTRANCE_WEAKER_THAN_BEST") or "0.032").strip())
    except ValueError:
        wk = 0.032
    wk = max(0.015, min(0.18, wk))

    survivors: list[dict] = []
    dropped = 0
    for d in ent:
        if id(d) == best_id:
            survivors.append(d)
            continue
        cf = float(d.get("confidence", 0))
        cy = (d["bbox"]["ymin"] + d["bbox"]["ymax"]) / (2.0 * max(img_h, 1))
        foreground_weaker = cy > cy_b + cy_slop and cf + 1e-9 < cb - wk
        far_below = cf + 1e-9 < cb - gap
        if far_below or foreground_weaker:
            dropped += 1
            continue
        survivors.append(d)

    if dropped == 0:
        return detections
    others = [d for d in detections if not _is_entrance_label(d.get("label"))]
    logger.info(
        "SAM3 streetview: pruned %d entrance(s) vs best (best_conf=%.3f cy_b=%.3f gap=%.2f)",
        dropped,
        cb,
        cy_b,
        gap,
    )
    return others + survivors


def _merge_sidewalk_detections(detections: list[dict], img_h: int) -> list[dict]:
    # Keep at most 2 sidewalk detections (largest by area).
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
    "building": 300,
    "roof": 300,
    "house": 300,
    "structure": 300,
    "building footprint": 300,
    "door": 3,
    "entrance": 8,
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
    # Keep only top N detections per class by confidence.
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
    "door": 280,
    "revolving door": 280,
    "glass entrance": 280,
    "storefront entrance": 280,
    "building entrance": 280,
    "person": 500,
    # Allow smaller building footprints so small houses and sheds are kept.
    "building": 70,
    "roof": 70,
    "house": 70,
    "structure": 70,
    "building footprint": 70,
}


def _min_area(bbox: dict, label: str = "", min_pixels: int = 1500) -> bool:
    w = bbox["xmax"] - bbox["xmin"]
    h = bbox["ymax"] - bbox["ymin"]
    threshold = _MIN_AREA_BY_LABEL.get(label, min_pixels)
    return w * h >= threshold


def _tensor_batch_len(x: Any) -> int:
    # Number of instances along dim 0 (boxes/scores/masks from SAM3 post-process).
    if x is None:
        return 0
    if hasattr(x, "shape") and len(getattr(x, "shape", ())) > 0:
        return int(x.shape[0])
    return len(x)


def _to_float_score(x: Any) -> float:
    if hasattr(x, "detach"):
        return float(x.detach().cpu().item())
    return float(x)


def _xyxy_from_box(box: Any) -> tuple[float, float, float, float]:
    # Normalize box to 4 floats whether it is a tensor slice, ndarray, or nested list.
    if hasattr(box, "detach"):
        flat = box.detach().cpu().flatten().tolist()
    elif hasattr(box, "tolist"):
        flat = box.tolist()
    else:
        flat = list(box)  # type: ignore[arg-type]
    while len(flat) == 1 and isinstance(flat[0], (list, tuple)):
        flat = list(flat[0])
    if len(flat) != 4:
        raise ValueError(f"expected 4 box coordinates, got {flat!r}")
    return float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])


def _clip_polygon_to_bounds(pts: list[list[float]], img_w: int, img_h: int) -> list[list[float]] | None:
    # Clip polygon points to image bounds so outlines stay inside the frame.
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
    # Shared mask preprocessing: squeeze, threshold, crop, upsample, morphology.
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
    # Convert a single OpenCV contour to a clipped polygon.
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
    # Extract the single largest polygon contour from binary mask (street view mode).
    arr, mh, mw = _prepare_mask(mask, img_w, img_h)
    if arr is None:
        return None
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    return _contour_to_polygon(cnt, mw, mh, img_w, img_h)


_SAT_MIN_CONTOUR_AREA = 80  # minimum contour area in pixels for satellite buildings


def _mask_to_all_polygons(mask, img_w: int, img_h: int) -> list[dict]:
    # Extract ALL contours from a mask as separate polygons (satellite mode).
    # Returns list of {polygon, bbox} dicts — one per detected building.
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


def _run_inference_pass(
    infer_image: Image.Image,
    prompts: list[str],
    infer_w: int,
    infer_h: int,
    confidence_threshold: float,
    mask_threshold: float,
    mode: str,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    *,
    batch_size: int | None = None,
) -> list[dict]:
    # Run one SAM 3 inference pass for a given image crop and concept prompt list.
    
    # Inputs:
    # - `infer_image`: PIL image crop to run inference on
    # - `prompts`: text concepts to evaluate (street view vs satellite differs)
    # - `infer_w`/`infer_h`: dimensions of the crop in pixels
    # - `offset_x`/`offset_y`: where this crop sits within the full image
    # - `scale_x`/`scale_y`: mapping from crop space back to full-image space
    
    # Output:
    # - list of dict detections containing:
    # - `label`: the prompt/concept that produced this instance
    # - `confidence`: score from SAM 3 post-processing
    # - `bbox`: bounding box mapped into full-image coordinates
    # - `polygon`: polygon outline derived from the segmentation mask (when available)
    import torch

    dets: list[dict] = []
    bs = batch_size if batch_size is not None else _BATCH_SIZE

    for batch_start in range(0, len(prompts), bs):
        batch_prompts = prompts[batch_start : batch_start + bs]
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

            try:
                cool_ms = int((os.environ.get("SAM3_BATCH_COOLDOWN_MS") or "0").strip())
            except ValueError:
                cool_ms = 0
            if cool_ms > 0:
                time.sleep(min(3000, cool_ms) / 1000.0)

            results = _processor.post_process_instance_segmentation(
                outputs,
                threshold=confidence_threshold,
                mask_threshold=mask_threshold,
                target_sizes=target_sizes,
            )

            for prompt, result in zip(batch_prompts, results):
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                masks = result.get("masks", [])

                n_inst = min(_tensor_batch_len(boxes), _tensor_batch_len(scores))
                n_masks = _tensor_batch_len(masks)

                for i in range(n_inst):
                    score_f = _to_float_score(scores[i])
                    if score_f < confidence_threshold:
                        continue

                    if mode == "satellite" and i < n_masks:
                        mask_arr = masks[i]
                        if hasattr(mask_arr, "cpu"):
                            mask_arr = mask_arr.cpu().numpy()
                        sub_polys = _mask_to_all_polygons(mask_arr, infer_w, infer_h)
                        for sp in sub_polys:
                            sb = sp["bbox"]
                            sb = {
                                "xmin": sb["xmin"] * scale_x + offset_x,
                                "ymin": sb["ymin"] * scale_y + offset_y,
                                "xmax": sb["xmax"] * scale_x + offset_x,
                                "ymax": sb["ymax"] * scale_y + offset_y,
                            }
                            poly = [
                                [px * scale_x + offset_x, py * scale_y + offset_y]
                                for px, py in sp["polygon"]
                            ]
                            dets.append({
                                "label": prompt,
                                "confidence": score_f,
                                "bbox": sb,
                                "polygon": poly,
                            })
                        continue

                    try:
                        x1, y1, x2, y2 = _xyxy_from_box(boxes[i])
                    except (ValueError, TypeError) as err:
                        logger.warning("Bad box tensor for prompt %r: %s", prompt, err)
                        continue
                    bbox = {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}
                    if not _min_area(bbox, label=prompt):
                        continue
                    polygon = None
                    # Default: skip mask→contour on street view (large CPU savings). Set SAM3_KEEP_MASK_POLYGON=1 for SAM outlines.
                    skip_poly = mode == "streetview" and not _env_truthy("SAM3_KEEP_MASK_POLYGON")
                    if not skip_poly and i < n_masks:
                        mask_arr = masks[i]
                        if hasattr(mask_arr, "cpu"):
                            mask_arr = mask_arr.cpu().numpy()
                        polygon = _mask_to_polygon(mask_arr, infer_w, infer_h)
                    bbox = {
                        "xmin": bbox["xmin"] * scale_x + offset_x,
                        "ymin": bbox["ymin"] * scale_y + offset_y,
                        "xmax": bbox["xmax"] * scale_x + offset_x,
                        "ymax": bbox["ymax"] * scale_y + offset_y,
                    }
                    if polygon:
                        polygon = [
                            [px * scale_x + offset_x, py * scale_y + offset_y]
                            for px, py in polygon
                        ]
                    dets.append({
                        "label": prompt,
                        "confidence": score_f,
                        "bbox": bbox,
                        "polygon": polygon,
                    })
        except Exception as e:
            logger.warning("SAM3 batch failed prompts=%s: %s", batch_prompts, e, exc_info=True)
            continue

    return dets


def _generate_tiles(w: int, h: int, tile_size: int, overlap: float = 0.25):
    # Yield (x, y, crop_w, crop_h) tiles covering the full image with overlap.
    step = int(tile_size * (1 - overlap))
    for y in range(0, h, step):
        for x in range(0, w, step):
            cw = min(tile_size, w - x)
            ch = min(tile_size, h - y)
            if cw < tile_size * 0.4 or ch < tile_size * 0.4:
                continue
            yield x, y, cw, ch


def _satellite_tiles_cover(w: int, h: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Axis-aligned windows of size up to tile_size with overlap; last step snaps to far edge."""
    if w <= 0 or h <= 0:
        return []
    ts = max(128, min(tile_size, max(w, h)))
    if w <= ts and h <= ts:
        return [(0, 0, w, h)]
    step = max(1, int(ts * (1 - overlap)))

    def axis_starts(total: int) -> list[int]:
        if total <= ts:
            return [0]
        xs: list[int] = []
        x = 0
        while True:
            xs.append(x)
            if x + ts >= total:
                break
            x = min(x + step, total - ts)
        return list(dict.fromkeys(xs))  # preserve order, drop dupes

    x_starts = axis_starts(w)
    y_starts = axis_starts(h)
    tiles: list[tuple[int, int, int, int]] = []
    for y in y_starts:
        for x in x_starts:
            cw = min(ts, w - x)
            ch = min(ts, h - y)
            tiles.append((x, y, cw, ch))
    return tiles


def _collect_satellite_detections(
    image: Image.Image,
    w: int,
    h: int,
    prompts: list[str],
    confidence_threshold: float,
    mask_threshold: float,
    max_dim: int,
) -> list[dict]:
    """Single downscaled pass, or optional multi-tile inference at ~max_dim per tile (SAM3_SAT_TILING=1)."""
    use_tiles = _env_truthy("SAM3_SAT_TILING")
    if not use_tiles or max(w, h) <= max_dim:
        infer_image = image
        sx, sy = 1.0, 1.0
        iw, ih = w, h
        if max(iw, ih) > max_dim:
            ratio = max_dim / max(iw, ih)
            iw, ih = int(iw * ratio), int(ih * ratio)
            infer_image = image.resize((iw, ih), Image.Resampling.BILINEAR)
            sx, sy = w / iw, h / ih
        logger.info("Satellite: single pass %dx%d (max_dim=%d)", iw, ih, max_dim)
        return _run_inference_pass(
            infer_image,
            prompts,
            iw,
            ih,
            confidence_threshold,
            mask_threshold,
            "satellite",
            scale_x=sx,
            scale_y=sy,
        )

    try:
        ov = float((os.environ.get("SAM3_SAT_TILE_OVERLAP") or "0.22").strip())
    except ValueError:
        ov = 0.22
    ov = max(0.08, min(0.45, ov))
    tile_specs = _satellite_tiles_cover(w, h, max_dim, ov)
    logger.info(
        "Satellite: tiled inference %d tiles max_dim=%d overlap=%.2f",
        len(tile_specs),
        max_dim,
        ov,
    )
    acc: list[dict] = []
    for x0, y0, cw, ch in tile_specs:
        crop = image.crop((x0, y0, x0 + cw, y0 + ch))
        iw, ih = crop.size
        infer_im = crop
        sx, sy = 1.0, 1.0
        if max(iw, ih) > max_dim:
            ratio = max_dim / max(iw, ih)
            ni, nj = max(1, int(iw * ratio)), max(1, int(ih * ratio))
            infer_im = crop.resize((ni, nj), Image.Resampling.BILINEAR)
            sx, sy = cw / ni, ch / nj
        else:
            ni, nj = iw, ih
        acc.extend(
            _run_inference_pass(
                infer_im,
                prompts,
                ni,
                nj,
                confidence_threshold,
                mask_threshold,
                "satellite",
                offset_x=float(x0),
                offset_y=float(y0),
                scale_x=sx,
                scale_y=sy,
            )
        )
    return acc


def _filter_satellite_vehicle_like_false_positives(
    detections: list[dict],
    img_w: int,
    img_h: int,
) -> list[dict]:
    """
    Parking-lot cars seen from above are small, elongated rectangles; SAM sometimes
    labels them "building". Drop tiny specks and small high-aspect boxes; tune via env.
    """
    if not detections or _env_truthy("SAM3_SAT_SKIP_VEHICLE_FILTER"):
        return detections

    img_area = float(max(1, img_w * img_h))
    try:
        min_area_frac = float((os.environ.get("SAM3_SAT_MIN_BUILDING_AREA_FRAC") or "0.00045").strip())
    except ValueError:
        min_area_frac = 0.00045
    min_area_frac = max(0.00012, min(0.0025, min_area_frac))

    try:
        elong_max_frac = float((os.environ.get("SAM3_SAT_ELONG_MAX_FRAC") or "0.0022").strip())
    except ValueError:
        elong_max_frac = 0.0022
    elong_max_frac = max(min_area_frac * 1.5, min(0.01, elong_max_frac))

    try:
        min_aspect = float((os.environ.get("SAM3_SAT_CARLIKE_ASPECT") or "2.12").strip())
    except ValueError:
        min_aspect = 2.12
    min_aspect = max(1.75, min(4.0, min_aspect))

    # Compact parking-lot flecks: both sides small vs frame and total area modest (cars, carts).
    try:
        fleck_max_frac = float((os.environ.get("SAM3_SAT_FLECK_MAX_FRAC") or "0.00095").strip())
    except ValueError:
        fleck_max_frac = 0.00095
    fleck_max_frac = max(0.0004, min(0.003, fleck_max_frac))
    try:
        fleck_cap_ratio = float((os.environ.get("SAM3_SAT_FLECK_MAX_SIDE_RATIO") or "0.052").strip())
    except ValueError:
        fleck_cap_ratio = 0.052
    fleck_cap_ratio = max(0.035, min(0.09, fleck_cap_ratio))
    short_side = float(min(img_w, img_h))
    fleck_px = fleck_cap_ratio * short_side

    kept: list[dict] = []
    dropped = 0
    for d in detections:
        b = d["bbox"]
        bw = max(0.0, float(b["xmax"]) - float(b["xmin"]))
        bh = max(0.0, float(b["ymax"]) - float(b["ymin"]))
        area = bw * bh
        frac = area / img_area
        if frac < min_area_frac:
            dropped += 1
            continue
        if frac <= fleck_max_frac and bw <= fleck_px + 0.5 and bh <= fleck_px + 0.5:
            dropped += 1
            continue
        short = min(bw, bh)
        long = max(bw, bh)
        aspect = long / max(short, 1e-6)
        if frac <= elong_max_frac and aspect >= min_aspect:
            dropped += 1
            continue
        kept.append(d)
    if dropped:
        logger.info(
            "SAM3 satellite: dropped %d vehicle-like/speck detections "
            "(min_area_frac=%.5f fleck_max_frac=%.5f side<=%.1fpx elong_max_frac=%.4f aspect>=%.2f)",
            dropped,
            min_area_frac,
            fleck_max_frac,
            fleck_px,
            elong_max_frac,
            min_aspect,
        )
    return kept


def run_detection(image_bytes: bytes, mode: str = "streetview") -> dict:
    # Run SAM 3 detection on image. Returns dict compatible with DetectionResult:
    # { image_width, image_height, detections, processing_time_s }
    
    # mode: "streetview" for door detection, "satellite" for building detection.
    # Satellite: one downscaled pass by default; set SAM3_SAT_TILING=1 for overlapping tiles.
    if not load_sam3():
        raise RuntimeError("SAM 3 model not loaded")

    start = time.perf_counter()
    with _sam3_inference_scope():
        return _run_detection_inner(image_bytes, mode, start)


def _run_detection_inner(image_bytes: bytes, mode: str, start: float) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    all_dets: list[dict] = []

    if mode == "satellite":
        prompts = SATELLITE_PROMPTS
        try:
            confidence_threshold = float((os.environ.get("SAM3_SAT_CONF") or "0.20").strip())
        except ValueError:
            confidence_threshold = 0.20
        confidence_threshold = max(0.12, min(0.45, confidence_threshold))
        try:
            mask_threshold = float((os.environ.get("SAM3_SAT_MASK_THRESHOLD") or "0.42").strip())
        except ValueError:
            mask_threshold = 0.42
        mask_threshold = max(0.25, min(0.65, mask_threshold))

        try:
            max_dim = int((os.environ.get("SAM3_SAT_MAX_DIM") or "768").strip())
        except ValueError:
            max_dim = 768
        max_dim = max(480, min(1024, max_dim))

        all_dets = _collect_satellite_detections(
            image, w, h, prompts, confidence_threshold, mask_threshold, max_dim
        )

        # Merge labels to "building"
        for d in all_dets:
            if d["label"] != "building":
                d["label"] = "building"

        # Drop only near–full-frame false positives (was 12% max, which deleted large real roofs).
        try:
            max_bbox_frac = float((os.environ.get("SAM3_SAT_MAX_BBOX_FRACTION") or "0.62").strip())
        except ValueError:
            max_bbox_frac = 0.62
        max_bbox_frac = max(0.18, min(0.92, max_bbox_frac))
        img_area = w * h
        all_dets = [
            d
            for d in all_dets
            if (d["bbox"]["xmax"] - d["bbox"]["xmin"]) * (d["bbox"]["ymax"] - d["bbox"]["ymin"])
            <= max_bbox_frac * img_area
        ]
        all_dets = _nms(all_dets, iou_threshold=0.6)
        all_dets = _filter_satellite_vehicle_like_false_positives(all_dets, w, h)

    else:
        try:
            max_dim = int(
                (os.environ.get("SAM3_STREET_MAX_DIM") or str(_DEFAULT_STREET_MAX_DIM)).strip()
            )
        except ValueError:
            max_dim = _DEFAULT_STREET_MAX_DIM
        max_dim = max(320, min(640, max_dim))

        try:
            veh_infer_max = int(
                (os.environ.get("SAM3_VEHICLE_INFER_MAX_DIM") or str(_DEFAULT_VEHICLE_INFER_MAX_DIM)).strip()
            )
        except ValueError:
            veh_infer_max = _DEFAULT_VEHICLE_INFER_MAX_DIM
        veh_infer_max = max(256, min(448, veh_infer_max))

        try:
            confidence_threshold = float((os.environ.get("SAM3_STREET_CONF") or "0.30").strip())
        except ValueError:
            confidence_threshold = 0.30
        confidence_threshold = max(0.15, min(0.50, confidence_threshold))

        street_mask_threshold = 0.45

        # Pass 1 — entrances at full working resolution (2 prompts, one forward).
        infer_image = image
        sx, sy = 1.0, 1.0
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            iw, ih = int(w * ratio), int(h * ratio)
            infer_image = image.resize((iw, ih), Image.Resampling.BILINEAR)
            sx, sy = w / iw, h / ih
        else:
            iw, ih = w, h

        logger.info(
            "Streetview pass1 entrances %dx%d conf=%.2f",
            iw,
            ih,
            confidence_threshold,
        )
        ent_dets = _run_inference_pass(
            infer_image,
            STREETVIEW_PROMPTS,
            iw,
            ih,
            confidence_threshold,
            street_mask_threshold,
            mode,
            scale_x=sx,
            scale_y=sy,
            batch_size=_BATCH_SIZE,
        )
        ent_dets = _nms(ent_dets, iou_threshold=0.6)

        all_dets = ent_dets
        if _street_vehicle_pass_enabled():
            try:
                gap_ms = int((os.environ.get("SAM3_PASS_GAP_MS") or "0").strip())
            except ValueError:
                gap_ms = 0
            if gap_ms > 0:
                time.sleep(min(3000, gap_ms) / 1000.0)
            vehicle_conf = max(0.12, min(0.35, confidence_threshold - 0.14))
            vr = veh_infer_max / max(w, h, 1)
            vw, vh = max(1, int(w * vr)), max(1, int(h * vr))
            veh_image = image.resize((vw, vh), Image.Resampling.BILINEAR)
            vsx, vsy = w / vw, h / vh
            logger.info(
                "Streetview pass2 vehicles %dx%d conf=%.2f",
                vw,
                vh,
                vehicle_conf,
            )
            veh_dets = _run_inference_pass(
                veh_image,
                STREETVIEW_VEHICLE_PROMPTS,
                vw,
                vh,
                vehicle_conf,
                street_mask_threshold,
                mode,
                scale_x=vsx,
                scale_y=vsy,
                batch_size=_BATCH_SIZE,
            )
            veh_dets = _nms(veh_dets, iou_threshold=0.55)
            all_dets = ent_dets + veh_dets

        all_dets = _nms_streetview_split_vehicle(all_dets)
        all_dets = _filter_person_building_overlap(all_dets, w, h)
        all_dets = _filter_entrances_on_vehicles(all_dets, w, h)
        all_dets = _filter_non_building_doors(all_dets, w, h)
        all_dets = _filter_street_entrance_foreground_clutter(all_dets, w, h)
        all_dets = _merge_entrance_detections(all_dets)
        try:
            y_lo = float((os.environ.get("SAM3_STREET_GROUND_DOOR_MIN_Y") or "0.38").strip())
        except ValueError:
            y_lo = 0.38
        try:
            y_hi = float((os.environ.get("SAM3_STREET_GROUND_DOOR_MAX_Y") or "0.50").strip())
        except ValueError:
            y_hi = 0.50
        y_lo = max(0.22, min(0.52, y_lo))
        y_hi = max(0.50, min(0.88, y_hi))
        if y_hi <= y_lo + 0.06:
            y_hi = y_lo + 0.06
        all_dets = _filter_first_floor_entrances(
            all_dets,
            h,
            min_center_y_ratio=y_lo,
            max_center_y_ratio=y_hi,
        )
        all_dets = _prefer_facade_entrance_cluster(all_dets, h)
        all_dets = _drop_entrances_when_only_foreground_band(all_dets, h)
        all_dets = _drop_entrances_far_below_best_confidence(all_dets, h)
        # Vehicle prompts are only for filtering; do not return car/truck to the client.
        all_dets = [
            d for d in all_dets
            if str(d.get("label", "")).strip().lower() not in _VEHICLE_LABELS
        ]
        # One primary entrance per street frame (main facade door). SAM3_MULTI_STREET_ENTRANCE=1 restores all.
        if not _env_truthy("SAM3_MULTI_STREET_ENTRANCE"):
            all_dets = _select_single_best_street_entrance(all_dets, w, h)

    all_dets = _cap_per_class(all_dets)
    if mode == "streetview" and not _env_truthy("SAM3_MULTI_STREET_ENTRANCE"):
        ent_only = [d for d in all_dets if _is_entrance_label(d.get("label"))]
        non_ent = [d for d in all_dets if not _is_entrance_label(d.get("label"))]
        if len(ent_only) > 1:
            best = max(ent_only, key=lambda d: _street_entrance_quality_score(d, w, h))
            all_dets = non_ent + [best]

    elapsed_s = round(time.perf_counter() - start, 3)

    detections = []
    for i, d in enumerate(all_dets):
        label = d["label"]
        # Normalize any entrance-like label to a single canonical label.
        if label in _ENTRANCE_LABELS:
            label = "entrance"
        det = {
            "id": f"det_{i}",
            "label": label,
            "confidence": d["confidence"],
            "bbox": d["bbox"],
        }
        if d.get("polygon"):
            det["polygon"] = d["polygon"]
        detections.append(det)

    if mode == "streetview" and not _env_truthy("SAM3_MULTI_STREET_ENTRANCE"):
        detections = _drop_entrances_far_below_best_confidence(detections, h)
        detections = _collapse_api_entrances_to_one(detections, w, h)

    logger.info(
        "Detection complete: %d objects in %.3fs (%s)",
        len(detections),
        elapsed_s,
        mode,
    )

    return {
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "processing_time_s": elapsed_s,
        "engine": "sam3",
    }
