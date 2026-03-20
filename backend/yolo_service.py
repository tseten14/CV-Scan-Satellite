# Ultralytics YOLO — prefers YOLO-World (text prompts → entrances/buildings) when local *world*.pt exists;
# falls back to YOLOv8 COCO if World/CLIP is unavailable. All weights loaded from disk only (no hub download).
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from sam3_service import _nms, _cap_per_class

logger = logging.getLogger("uvicorn.error")

_backend_dir = Path(__file__).resolve().parent

# Open-vocabulary — keep prompt count modest (many classes can hurt YOLO-World quality).
# SAM 3 parity + one extra residential phrase.
STREETVIEW_CLASSES = [
    "door",
    "front door",
    "revolving door",
    "glass entrance",
    "storefront entrance",
    "building entrance",
    "car",
    "truck",
]

SATELLITE_CLASSES = [
    "building",
    "roof",
    "house",
    "structure",
    "building footprint",
]

_ENTRANCE_LABELS = frozenset(
    {
        "door",
        "front door",
        "revolving door",
        "glass entrance",
        "storefront entrance",
        "building entrance",
        "entrance",
    }
)


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes")


def _is_tiny_street_image(iw: int, ih: int) -> bool:
    """html2canvas map strips are often ~200–400px — YOLO-World scores are low; relax filters."""
    long_side = max(iw, ih, 1)
    return long_side < int(_float_env("YOLO_TINY_IMAGE_MAX_SIDE", 450)) or (iw * ih) < int(
        _float_env("YOLO_TINY_IMAGE_MAX_PIXELS", 110_000)
    )


def _drop_extreme_pencil_entrances(
    dets: list[dict], iw: int, ih: int, *, tiny_image: bool = False
) -> list[dict]:
    """Default-on minimal rule: only obvious sidelite strips (two-tier, avoids killing real doors)."""
    rw_a = _float_env("YOLO_PENCIL_MAX_RW_A", 0.034)
    ar_a = _float_env("YOLO_PENCIL_MIN_AR_A", 3.02)
    rw_b = _float_env("YOLO_PENCIL_MAX_RW_B", 0.044)
    ar_b = _float_env("YOLO_PENCIL_MIN_AR_B", 3.42)
    # Wider-but-still-tall porch window (left of front door); needs modest AR, not pencil-thin.
    rw_c = _float_env("YOLO_PENCIL_MAX_RW_C", 0.056)
    ar_c = _float_env("YOLO_PENCIL_MIN_AR_C", 2.18)
    conf_c = _float_env("YOLO_PENCIL_MAX_CONF_C", 0.52)
    out: list[dict] = []
    for d in dets:
        if _normalize_label(d["label"]) != "entrance":
            out.append(d)
            continue
        b = d["bbox"]
        bw = max(0.0, b["xmax"] - b["xmin"])
        bh = max(0.0, b["ymax"] - b["ymin"])
        rw = bw / max(iw, 1)
        ar = bh / max(bw, 1e-6)
        conf = float(d.get("confidence", 0.0))
        if rw < rw_a and ar > ar_a:
            continue
        if rw < rw_b and ar > ar_b:
            continue
        if not tiny_image and rw < rw_c and ar > ar_c and conf <= conf_c:
            continue
        out.append(d)
    return out


def _filter_weak_or_sidelike_entrances(
    dets: list[dict], iw: int, ih: int, *, tiny_image: bool = False
) -> list[dict]:
    """
    Drop entrance boxes that are almost certainly noise:
    - very low model score (e.g. 6% — not trustworthy)
    - narrow + tall + weak score (porch sidelight / side window, not a full door)
    """
    min_show = _float_env("YOLO_ENTRANCE_MIN_DISPLAY_CONF", 0.11)
    if tiny_image:
        min_show = min(min_show, _float_env("YOLO_LOWRES_MIN_DISPLAY_CONF", 0.034))
    nrw = _float_env("YOLO_WEAK_SIDEWINDOW_MAX_RW", 0.064)
    nar = _float_env("YOLO_WEAK_SIDEWINDOW_MIN_AR", 1.9)
    nconf = _float_env("YOLO_WEAK_SIDEWINDOW_MAX_CONF", 0.48)
    out: list[dict] = []
    for d in dets:
        if _normalize_label(d["label"]) != "entrance":
            out.append(d)
            continue
        conf = float(d.get("confidence", 0.0))
        if conf < min_show:
            continue
        b = d["bbox"]
        bw = max(0.0, b["xmax"] - b["xmin"])
        bh = max(0.0, b["ymax"] - b["ymin"])
        rw = bw / max(iw, 1)
        ar = bh / max(bw, 1e-6)
        # Tiny map captures: scores are compressed — don't drop "narrow" heuristically.
        if not tiny_image and rw < nrw and ar >= nar and conf <= nconf:
            continue
        out.append(d)
    return out


def _filter_distant_micro_entrances(
    dets: list[dict], iw: int, ih: int, *, tiny_image: bool = False
) -> list[dict]:
    """
    Full-res Street View: model often returns one tiny 'entrance' on a far alley / garage
    while missing main doors. Drop micro boxes unless confidence is high enough to trust.
    Skipped for tiny map captures (different scale semantics).
    """
    if tiny_image:
        return dets
    min_area = _float_env("YOLO_DISTANT_MICRO_ENTRANCE_AREA_FRAC", 0.0009)
    min_conf_micro = _float_env("YOLO_MICRO_ENTRANCE_MIN_CONF", 0.5)
    min_rh = _float_env("YOLO_MICRO_ENTRANCE_MIN_REL_HEIGHT", 0.024)
    top_frac = _float_env("YOLO_DISTANT_ENTRANCE_TOP_FRAC", 0.36)
    hi_conf = _float_env("YOLO_HIGH_BAND_ENTRANCE_MIN_CONF", 0.46)

    out: list[dict] = []
    for d in dets:
        if _normalize_label(d["label"]) != "entrance":
            out.append(d)
            continue
        b = d["bbox"]
        bw = max(0.0, b["xmax"] - b["xmin"])
        bh = max(0.0, b["ymax"] - b["ymin"])
        area_frac = (bw * bh) / max(iw * ih, 1)
        rh = bh / max(ih, 1)
        conf = float(d.get("confidence", 0.0))
        cy = 0.5 * (b["ymin"] + b["ymax"])

        is_micro = area_frac < min_area or rh < min_rh
        if is_micro and conf < min_conf_micro:
            continue
        if area_frac < min_area * 2.5 and cy < ih * top_frac and conf < hi_conf:
            continue
        out.append(d)
    return out


# Street View UI (chevrons, overlays) is often mislabeled by COCO / open-vocab (e.g. "airplane").
_STREETVIEW_SUPPRESSED_LABELS = frozenset(
    {
        "airplane",
        "kite",
        "frisbee",
    }
)

_WORLD_WEIGHTS = (
    "yolov8s-worldv2.pt",
    "yolov8s-world.pt",
    "yolov8m-worldv2.pt",
)

_STANDARD_WEIGHTS = (
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
)

YoloVariant = Literal["world", "coco"]

_yolo_model: Any = None
_loaded_weights: str | None = None
_yolo_variant: YoloVariant | None = None

_yolo_ssl_prepared = False
_yolo_unverified_https_applied = False


def _is_likely_ssl_verify_failure(exc: BaseException) -> bool:
    parts: list[str] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{cur.__class__.__name__}: {cur}".lower())
        cur = cur.__cause__ or cur.__context__  # type: ignore[assignment]
    blob = " ".join(parts)
    return "certificate verify failed" in blob or "certificate_verify_failed" in blob


def _relax_https_verify_globally() -> None:
    """Last resort for corporate SSL inspection (urllib / torch hub / CLIP)."""
    global _yolo_unverified_https_applied
    if _yolo_unverified_https_applied:
        return
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: S506
    _yolo_unverified_https_applied = True
    logger.warning(
        "HTTPS certificate verification is DISABLED for this process (YOLO / CLIP downloads). "
        "Prefer fixing trust: set SSL_CERT_FILE to your org CA bundle, or use YOLO_PREFER_COCO=1 with local yolov8n.pt only."
    )


def _prepare_ssl_for_yolo() -> None:
    """
    YOLO-World + CLIP often trigger urllib/torch HTTPS downloads. Corporate proxies break verify unless
    SSL_CERT_FILE includes the inspection CA — many Mac/Python installs also miss the certifi bundle.
    """
    global _yolo_ssl_prepared
    if _yolo_ssl_prepared:
        return
    _yolo_ssl_prepared = True

    if not (os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")):
        try:
            import certifi

            bundle = certifi.where()
            os.environ.setdefault("SSL_CERT_FILE", bundle)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
            logger.info("YOLO: using certifi CA bundle for HTTPS (SSL_CERT_FILE unset).")
        except ImportError:
            logger.debug("certifi not installed; skipping default CA bundle for YOLO.")

    insecure = (
        os.environ.get("YOLO_INSECURE_SSL") or os.environ.get("CVSCAN_INSECURE_SSL") or ""
    ).strip().lower()
    if insecure in ("1", "true", "yes"):
        _relax_https_verify_globally()


def _is_world_checkpoint(path: Path) -> bool:
    return "world" in path.name.lower()


def _weight_candidates() -> list[tuple[Path, YoloVariant]]:
    """Ordered (path, variant) to try. Respects YOLO_WEIGHTS, YOLO_PREFER_COCO."""
    prefer_coco = (os.environ.get("YOLO_PREFER_COCO") or "").strip() in ("1", "true", "yes")
    env = (os.environ.get("YOLO_WEIGHTS") or "").strip()
    out: list[tuple[Path, YoloVariant]] = []

    if env:
        p = Path(env).expanduser()
        if p.is_file():
            kind: YoloVariant = "world" if _is_world_checkpoint(p) else "coco"
            out.append((p.resolve(), kind))
        else:
            logger.error("YOLO_WEIGHTS points to missing file: %s", p)
        return out

    world_pairs = [(_backend_dir / n, "world") for n in _WORLD_WEIGHTS if (_backend_dir / n).is_file()]
    coco_pairs = [(_backend_dir / n, "coco") for n in _STANDARD_WEIGHTS if (_backend_dir / n).is_file()]

    if prefer_coco:
        out.extend(coco_pairs)
        out.extend(world_pairs)
    else:
        out.extend(world_pairs)
        out.extend(coco_pairs)

    return out


def load_yolo() -> bool:
    """Load YOLO-World or YOLOv8 once from local checkpoints. Returns True on success."""
    global _yolo_model, _loaded_weights, _yolo_variant
    _prepare_ssl_for_yolo()
    if _yolo_model is not None:
        return True

    candidates = _weight_candidates()
    if not candidates:
        logger.error(
            "No YOLO weights in %s. Add yolov8s-worldv2.pt (entrances) and/or yolov8n.pt — see README.",
            _backend_dir,
        )
        return False

    last_err: Exception | None = None
    for path, kind in candidates:
        try:
            if kind == "world":
                try:
                    from ultralytics import YOLOWorld as WorldModel  # type: ignore
                except ImportError:
                    WorldModel = None  # type: ignore
                if WorldModel is None:
                    raise RuntimeError("YOLOWorld not available in this ultralytics build")
                logger.info("Loading YOLO-World weights: %s …", path)
                candidate = WorldModel(str(path))
                if not hasattr(candidate, "set_classes"):
                    raise RuntimeError(f"{path.name} is not a YOLO-World checkpoint (no set_classes)")
                _yolo_model = candidate
                _loaded_weights = path.name
                _yolo_variant = "world"
                logger.info("YOLO-World ready (%s).", _loaded_weights)
                return True

            from ultralytics import YOLO  # type: ignore

            logger.info("Loading YOLOv8 (COCO) weights: %s …", path)
            _yolo_model = YOLO(str(path))
            _loaded_weights = path.name
            _yolo_variant = "coco"
            logger.info("YOLOv8 COCO ready (%s).", _loaded_weights)
            return True
        except Exception as e:
            last_err = e
            logger.warning("YOLO load failed (%s, %s): %s", path.name, kind, e)
            _yolo_model = None
            _loaded_weights = None
            _yolo_variant = None
            continue

    logger.error("All YOLO weight attempts failed. Last error: %s", last_err)
    return False


def _bbox_to_polygon(bbox: dict) -> list[list[float]]:
    x0, y0 = bbox["xmin"], bbox["ymin"]
    x1, y1 = bbox["xmax"], bbox["ymax"]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _normalize_label(raw: str) -> str:
    return (raw or "").strip().lower()


def _streetview_suppressed_labels() -> frozenset[str]:
    extra = (os.environ.get("YOLO_STREETVIEW_EXTRA_DROP_LABELS") or "").strip()
    if not extra:
        return _STREETVIEW_SUPPRESSED_LABELS
    more = {_normalize_label(x) for x in extra.split(",") if x.strip()}
    return frozenset(_STREETVIEW_SUPPRESSED_LABELS | more)


def _looks_like_streetview_nav_overlay(bbox: dict, iw: int, ih: int) -> bool:
    """
    Heuristic: small box, low in frame, roughly centered — typical Google Street View
    forward/back chevrons on the road (often misclassified as airplane etc.).
    """
    if iw <= 0 or ih <= 0:
        return False
    w_box = max(0.0, bbox["xmax"] - bbox["xmin"])
    h_box = max(0.0, bbox["ymax"] - bbox["ymin"])
    area_frac = (w_box * h_box) / (iw * ih)
    cy = 0.5 * (bbox["ymin"] + bbox["ymax"])
    cx = 0.5 * (bbox["xmin"] + bbox["xmax"])
    try:
        bottom_frac = float(os.environ.get("YOLO_STREETVIEW_UI_BOTTOM_FRAC", "0.14"))
    except ValueError:
        bottom_frac = 0.14
    bottom_frac = max(0.05, min(0.35, bottom_frac))
    y_min_keep = ih * (1.0 - bottom_frac)
    if cy < y_min_keep:
        return False
    if not (0.12 * iw < cx < 0.88 * iw):
        return False
    if area_frac < 0.0008 or area_frac > 0.045:
        return False
    ar = w_box / max(h_box, 1e-6)
    if ar < 0.25 or ar > 5.0:
        return False
    return True


def _filter_streetview_ui_false_positives(dets: list[dict], iw: int, ih: int) -> list[dict]:
    suppressed = _streetview_suppressed_labels()
    out: list[dict] = []
    for d in dets:
        lbl = _normalize_label(d["label"])
        if lbl in suppressed:
            continue
        if _looks_like_streetview_nav_overlay(d["bbox"], iw, ih):
            continue
        out.append(d)
    return out


def _float_env(name: str, default: float) -> float:
    try:
        return float((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _filter_implausible_yolo_entrances(
    dets: list[dict],
    iw: int,
    ih: int,
    *,
    strict: bool,
) -> list[dict]:
    """
    YOLO-World often fires on narrow vertical windows / sidelights as 'entrance'.
    `strict=True`: balanced precision/recall. `strict=False`: only drop obvious sidelite junk
    so we still show something when the model is weak (residential facades).
    """
    img_area = max(iw * ih, 1)
    if strict:
        min_area_frac = _float_env("YOLO_ENTRANCE_MIN_AREA_FRAC", 0.0012)
        max_area_frac = _float_env("YOLO_ENTRANCE_MAX_AREA_FRAC", 0.28)
        narrow_rw = _float_env("YOLO_ENTRANCE_NARROW_MAX_WIDTH_FRAC", 0.034)
        narrow_ar = _float_env("YOLO_ENTRANCE_NARROW_MIN_ASPECT", 2.45)
        min_conf = _float_env("YOLO_ENTRANCE_MIN_CONFIDENCE", 0.14)
        dubious_conf = _float_env("YOLO_ENTRANCE_DUBIOUS_CONFIDENCE", 0.38)
        min_h_frac = _float_env("YOLO_ENTRANCE_MIN_HEIGHT_FRAC", 0.022)
    else:
        min_area_frac = 0.00045
        max_area_frac = 0.35
        narrow_rw = 0.028
        narrow_ar = 2.65
        min_conf = 0.08
        dubious_conf = 0.25
        min_h_frac = 0.014

    out: list[dict] = []
    for d in dets:
        lbl = _normalize_label(d["label"])
        if lbl != "entrance":
            out.append(d)
            continue

        b = d["bbox"]
        bw = max(0.0, b["xmax"] - b["xmin"])
        bh = max(0.0, b["ymax"] - b["ymin"])
        area_frac = (bw * bh) / img_area
        conf = float(d.get("confidence", 0.0))
        rw = bw / max(iw, 1)
        rh = bh / max(ih, 1)
        ar = bh / max(bw, 1e-6)

        if conf < min_conf:
            continue
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        if rh < min_h_frac:
            continue
        # Tall narrow strip (sidelight / vertical window) — common false positive.
        if rw < narrow_rw and ar >= narrow_ar:
            continue
        if strict:
            if ar > 4.5 and conf < dubious_conf:
                continue
            if ar < 0.5 and area_frac < 0.008 and conf < dubious_conf:
                continue

        out.append(d)
    return out


def _filter_yolo_porch_sidelights(dets: list[dict], iw: int, ih: int) -> list[dict]:
    """
    Optional aggressive sidelight removal — opt-in (was default and killed valid doors).
    Enable with YOLO_SIDELIGHT_FILTER=1.
    """
    out: list[dict] = []
    rw_max = _float_env("YOLO_SIDELIGHT_MAX_RW", 0.062)
    ar_min = _float_env("YOLO_SIDELIGHT_MIN_AR", 1.88)
    conf_max = _float_env("YOLO_SIDELIGHT_MAX_CONF", 0.58)
    rw_hard = _float_env("YOLO_SIDELIGHT_HARD_MAX_RW", 0.051)
    ar_hard = _float_env("YOLO_SIDELIGHT_HARD_MIN_AR", 2.02)
    rw_extreme = _float_env("YOLO_SIDELIGHT_EXTREME_MAX_RW", 0.046)
    ar_extreme = _float_env("YOLO_SIDELIGHT_EXTREME_MIN_AR", 1.9)

    for d in dets:
        if _normalize_label(d["label"]) != "entrance":
            out.append(d)
            continue
        b = d["bbox"]
        bw = max(0.0, b["xmax"] - b["xmin"])
        bh = max(0.0, b["ymax"] - b["ymin"])
        rw = bw / max(iw, 1)
        ar = bh / max(bw, 1e-6)
        conf = float(d.get("confidence", 0.0))

        if rw < rw_extreme and ar >= ar_extreme:
            continue
        if rw < rw_hard and ar >= ar_hard:
            continue
        if rw < rw_max and ar >= ar_min and conf <= conf_max:
            continue

        out.append(d)
    return out


def run_yolo_detection(image_bytes: bytes, mode: str = "streetview") -> dict:
    """
    Run YOLO on image bytes. Same JSON shape as SAM 3 run_detection(), plus engine: \"yolo\"
    and yolo_variant: \"world\" | \"coco\".
    """
    _prepare_ssl_for_yolo()
    if not load_yolo():
        raise RuntimeError(
            "YOLO model not loaded. Place yolov8s-worldv2.pt (recommended for entrances) "
            "or yolov8n.pt in backend/, install openai-clip, and restart — see README."
        )

    assert _yolo_variant is not None
    variant: YoloVariant = _yolo_variant

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
    tiny = mode == "streetview" and _is_tiny_street_image(w, h)

    start = time.perf_counter()
    long_side = max(w, h)
    # Upscale inference size for tiny html2canvas captures so the detector has enough context.
    if tiny:
        imgsz = int(min(1280, max(512, long_side * int(_float_env("YOLO_TINY_IMGSZ_MULT", 2)))))
        logger.info(
            "YOLO: tiny street image %sx%s — imgsz=%s, relaxed post-filters (prefer full Street View or scale-2+ capture)",
            w,
            h,
            imgsz,
        )
    else:
        imgsz = min(1280, long_side)

    if variant == "world":
        if mode == "satellite":
            conf = 0.08
        else:
            base = _float_env("YOLO_WORLD_STREET_CONF", 0.055)
            if tiny:
                conf = min(base, _float_env("YOLO_WORLD_STREET_CONF_TINY", 0.028))
            else:
                conf = base
    else:
        conf = 0.12 if mode == "satellite" else 0.25

    def _run_forward() -> Any:
        if variant == "world":
            _yolo_model.set_classes(
                SATELLITE_CLASSES if mode == "satellite" else STREETVIEW_CLASSES
            )
        return _yolo_model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            max_det=300,
            verbose=False,
        )

    try:
        results = _run_forward()
    except Exception as e:
        retry_ok = (
            os.environ.get("YOLO_RETRY_WITH_UNVERIFIED_SSL") or ""
        ).strip().lower() in ("1", "true", "yes")
        if retry_ok and _is_likely_ssl_verify_failure(e) and not _yolo_unverified_https_applied:
            logger.warning(
                "YOLO: SSL verify failed (%s); retrying once with unverified HTTPS "
                "(YOLO_RETRY_WITH_UNVERIFIED_SSL=1). Prefer SSL_CERT_FILE=… with your org CA.",
                e.__class__.__name__,
            )
            _relax_https_verify_globally()
            try:
                results = _run_forward()
            except Exception as e2:
                logger.exception("YOLO predict failed after SSL retry: %s", e2)
                raise RuntimeError(f"YOLO inference failed: {e2}") from e2
        else:
            logger.exception("YOLO predict failed: %s", e)
            if _is_likely_ssl_verify_failure(e):
                raise RuntimeError(
                    f"YOLO inference failed: {e}. "
                    "SSL: set SSL_CERT_FILE to your certificate bundle, or export YOLO_INSECURE_SSL=1 "
                    "(dev only), or YOLO_RETRY_WITH_UNVERIFIED_SSL=1 for a one-time retry, "
                    "or YOLO_PREFER_COCO=1 with local yolov8n.pt to skip CLIP."
                ) from e
            raise RuntimeError(f"YOLO inference failed: {e}") from e

    r = results[0]
    names = r.names if r.names is not None else {}
    dets: list[dict] = []

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        n = len(boxes)
        for i in range(n):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf_sc = float(boxes.conf[i].cpu().numpy())
            cls_i = int(boxes.cls[i].cpu().numpy())
            if isinstance(names, dict):
                raw_lbl = names.get(cls_i, str(cls_i))
            else:
                raw_lbl = names[cls_i] if cls_i < len(names) else str(cls_i)
            label = _normalize_label(str(raw_lbl))

            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            bbox = {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}
            dets.append({"label": label, "confidence": conf_sc, "bbox": bbox})

    if mode == "streetview":
        dets = _filter_streetview_ui_false_positives(dets, w, h)

    if mode == "satellite":
        for d in dets:
            d["label"] = "building"
        img_area = w * h
        dets = [
            d
            for d in dets
            if (d["bbox"]["xmax"] - d["bbox"]["xmin"]) * (d["bbox"]["ymax"] - d["bbox"]["ymin"])
            < 0.12 * img_area
        ]
        dets = _nms(dets, iou_threshold=0.6)
    elif variant == "world":
        for d in dets:
            if d["label"] in _ENTRANCE_LABELS:
                d["label"] = "entrance"
        if mode == "streetview":
            if _env_flag("YOLO_ENTRANCE_GEOMETRY_FILTER"):
                strict_on = not _env_flag("YOLO_ENTRANCE_STRICT_ONLY")
                dets_strict = _filter_implausible_yolo_entrances(dets, w, h, strict=True)
                has_ent = any(_normalize_label(x["label"]) == "entrance" for x in dets_strict)
                if has_ent or not strict_on:
                    dets = dets_strict
                else:
                    dets_loose = _filter_implausible_yolo_entrances(dets, w, h, strict=False)
                    if any(_normalize_label(x["label"]) == "entrance" for x in dets_loose):
                        logger.info(
                            "YOLO-World: strict entrance filter removed all boxes; using loose filter."
                        )
                        dets = dets_loose
                    else:
                        dets = dets_strict
            else:
                dets = _drop_extreme_pencil_entrances(dets, w, h, tiny_image=tiny)
            dets = _filter_weak_or_sidelike_entrances(dets, w, h, tiny_image=tiny)
            dets = _filter_distant_micro_entrances(dets, w, h, tiny_image=tiny)
            if _env_flag("YOLO_SIDELIGHT_FILTER"):
                dets = _filter_yolo_porch_sidelights(dets, w, h)
        nms_kw: dict = {"iou_threshold": 0.6}
        if (os.environ.get("YOLO_ENTRANCE_NMS_IOU") or "").strip():
            nms_kw["entrance_suppress_iou"] = _float_env("YOLO_ENTRANCE_NMS_IOU", 0.5)
        dets = _nms(dets, **nms_kw)
    else:
        dets = _nms(dets, iou_threshold=0.6)

    dets = _cap_per_class(dets)

    detections: list[dict] = []
    for i, d in enumerate(dets):
        det: dict = {
            "id": f"det_{i}",
            "label": d["label"],
            "confidence": d["confidence"],
            "bbox": d["bbox"],
            "polygon": _bbox_to_polygon(d["bbox"]),
        }
        detections.append(det)

    elapsed_s = round(time.perf_counter() - start, 3)
    logger.info("YOLO (%s): %s objects in %.3fs (%s)", variant, len(detections), elapsed_s, mode)

    return {
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "processing_time_s": elapsed_s,
        "engine": "yolo",
        "yolo_variant": variant,
    }


def get_yolo_variant() -> YoloVariant | None:
    """Which stack is loaded after a successful load_yolo() (None if not loaded)."""
    return _yolo_variant
