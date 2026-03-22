# FastAPI backend for scene detection: SAM 3 and YOLO (World + COCO fallback; compare via ?engine=sam3|yolo).
# Exposes /detect for uploaded images and /streetview for fetching street view imagery.
import logging
import math
import os
import httpx

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from sam3_service import run_detection, load_sam3
from yolo_service import run_yolo_detection
from entrances import get_entrances, get_cta_entrances

logger = logging.getLogger("uvicorn.error")


app = FastAPI(
    title="CV-SCAN-GEOAI Detection API",
    description="Scene detection via SAM 3 or YOLO (YOLO-World when local weights exist, else YOLOv8 COCO)",
)


@app.on_event("startup")
async def startup():
    # FastAPI startup hook.

    # We eagerly attempt to load the heavy SAM 3 model once when the server boots so
    # the first user request does not pay the model download/initialization cost.

    # If SAM 3 cannot be loaded (missing Hugging Face access, missing token, etc.),
    # we do *not* crash the server; endpoints will fail later with a clear error.
    try:
        load_sam3()
    except Exception as e:
        logger.warning(f"SAM 3 preload skipped: {e}")


app.add_middleware(
    # Allow the frontend (running on a different port) to call this API.
    # This is required for browser fetch() requests to /detect and /streetview-image.
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://127.0.0.1:5173", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Satellite-Bbox-West",
        "X-Satellite-Bbox-East",
        "X-Satellite-Bbox-North",
        "X-Satellite-Bbox-South",
        "X-Satellite-Center-Lat",
        "X-Satellite-Center-Lng",
    ],
)


@app.get("/health")
async def health():
    # Simple health check endpoint for debugging and monitoring.

    # This is intentionally lightweight and does not require the SAM 3 model.
    return {"status": "ok"}


_GMAPS_EMBED_KEY = "AIzaSyCmL18misQw9KdwqGaw3zHkitj8vG6QF2Y"
_HERE_OIS_KEY = (os.environ.get("HERE_OIS_API_KEY") or "").strip()


def _bbox_from_center(lat: float, lng: float, size_m: float) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) for a center point and square side length in meters."""
    radius_m = 6378137.0
    half = size_m / 2.0
    lat_off = (half / radius_m) * (180.0 / math.pi)
    lon_off = (half / (radius_m * math.cos(math.radians(lat)))) * (180.0 / math.pi)
    return (lng - lon_off, lat - lat_off, lng + lon_off, lat + lat_off)


@app.get("/streetview-image")
async def streetview_image(
    lat: float = Query(...),
    lng: float = Query(...),
    heading: float = Query(0),
):
    # Fetch a single Street View image facing toward the pin location.
    import math

    # User-Agent can help avoid some automated-request throttling from the provider.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        # We use the Google Street View metadata endpoint to resolve the panorama id (pano_id)
        # that is closest to the requested lat/lng.

        # Then we request a 640x640 thumbnail tile at the computed heading so that
        # the resulting image faces toward the selected pin.
        async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
            meta_url = (
                f"https://maps.googleapis.com/maps/api/streetview/metadata"
                f"?location={lat},{lng}&source=outdoor&key={_GMAPS_EMBED_KEY}"
            )
            meta_resp = await client.get(meta_url)
            if meta_resp.status_code != 200:
                raise HTTPException(502, "Street view metadata lookup failed")

            import json
            meta = json.loads(meta_resp.text)
            if meta.get("status") != "OK":
                raise HTTPException(
                    404,
                    "No street view panorama found near this location. "
                    "Try dropping the pin on a road.",
                )

            pano_id = meta["pano_id"]
            pano_lat = meta.get("location", {}).get("lat", lat)
            pano_lng = meta.get("location", {}).get("lng", lng)

            # Compute heading from panorama position toward the dropped pin
            d_lat = lat - pano_lat
            d_lng = lng - pano_lng
            if abs(d_lat) > 1e-7 or abs(d_lng) > 1e-7:
                face_heading = math.degrees(math.atan2(d_lng, d_lat)) % 360
            else:
                face_heading = heading

            logger.info(f"Resolved pano_id={pano_id}, heading={face_heading:.1f}° for ({lat}, {lng})")

            thumb_url = (
                f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
                f"?panoid={pano_id}"
                f"&cb_client=search.revgeo_and_hierarchicalsearch.geoname"
                f"&w=640&h=640"
                f"&yaw={face_heading}&pitch=0&thumbfov=90"
            )
            img_resp = await client.get(thumb_url)
            if img_resp.status_code != 200:
                raise HTTPException(502, "Failed to fetch street view image")

            content_type = img_resp.headers.get("content-type", "")
            if "image" not in content_type:
                raise HTTPException(502, "Street view returned non-image response")

            return Response(content=img_resp.content, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Street view fetch failed")
        raise HTTPException(502, f"Street view fetch failed: {e}")


@app.get("/satellite-image")
async def satellite_image(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    box_size_m: float = Query(220, ge=20, le=10000),
    width: int = Query(1200, ge=128, le=2048),
    height: int = Query(1200, ge=128, le=2048),
):
    """Fetch a HERE satellite image centered at lat/lng and return bbox metadata in headers."""
    if not _HERE_OIS_KEY:
        raise HTTPException(
            500,
            "HERE_OIS_API_KEY is not configured on the backend environment",
        )

    west, south, east, north = _bbox_from_center(lat, lng, box_size_m)
    bbox = f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"
    url = (
        "https://ois.had.in.here.com/api/rest/v1/wms/getMap/"
        f"bbox/{bbox}/width/{width}/height/{height}/format/png"
    )

    params = {
        "imageryType": "VEXCEL",
        "resolutionType": "LOW",
        "apiKey": _HERE_OIS_KEY,
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient(timeout=25, headers=headers, follow_redirects=True) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.warning("HERE satellite fetch failed %s: %s", resp.status_code, resp.text[:240])
                raise HTTPException(502, "Failed to fetch HERE satellite image")

            content_type = (resp.headers.get("content-type") or "").lower()
            if "image" not in content_type:
                logger.warning("HERE satellite non-image response: %s", resp.text[:240])
                raise HTTPException(502, "HERE satellite API returned non-image response")

            return Response(
                content=resp.content,
                media_type="image/png",
                headers={
                    "X-Satellite-Bbox-West": f"{west:.8f}",
                    "X-Satellite-Bbox-East": f"{east:.8f}",
                    "X-Satellite-Bbox-North": f"{north:.8f}",
                    "X-Satellite-Bbox-South": f"{south:.8f}",
                    "X-Satellite-Center-Lat": f"{lat:.8f}",
                    "X-Satellite-Center-Lng": f"{lng:.8f}",
                },
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("HERE satellite fetch failed")
        raise HTTPException(502, f"Satellite fetch failed: {e}")


# --- Venue Finder (transit entrances) — proxied from frontend as /api/entrances* ---


@app.get("/entrances")
def search_entrances(
    query: str = Query(..., min_length=1, description="Station or location name"),
    lat_min: float | None = Query(None, description="Bounding box lat min"),
    lat_max: float | None = Query(None, description="Bounding box lat max"),
    lon_min: float | None = Query(None, description="Bounding box lon min"),
    lon_max: float | None = Query(None, description="Bounding box lon max"),
):
    """GTFS-derived transit entrances (merged from venue-finder-ai backend)."""
    results = get_entrances(
        query=query,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    return {"entrances": results}


@app.get("/entrances/cta")
def cta_entrances(
    lat_min: float | None = Query(None, description="Bounding box lat min"),
    lat_max: float | None = Query(None, description="Bounding box lat max"),
    lon_min: float | None = Query(None, description="Bounding box lon min"),
    lon_max: float | None = Query(None, description="Bounding box lon max"),
):
    """Chicago CTA entrances from backend/data/entrances/cta.txt."""
    results = get_cta_entrances(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    return {"entrances": results}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    mode: str = Query("streetview", pattern="^(streetview|satellite)$"),
    engine: str = Query("sam3", pattern="^(sam3|yolo)$"),
):
    # Main detection endpoint.

    # The frontend sends an uploaded image (from Street View or a map screenshot/upload)
    # along with query parameters:
    #   - `mode`: streetview (entrances) | satellite (buildings)
    #   - `engine`: sam3 | yolo  (YOLO-World open-vocab if *world*.pt present, else COCO YOLOv8)

    # SAM 3: `run_detection()` in `sam3_service.py`.
    # YOLO: `run_yolo_detection()` in `yolo_service.py`.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpeg, png, webp)")

    try:
        # UploadFile is streamed by FastAPI; we read the bytes in memory
        # because SAM 3 expects image bytes that we can wrap in a PIL image.
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file")

    logger.info("POST /detect mode=%s engine=%s bytes=%s", mode, engine, len(image_bytes))

    try:
        if engine == "yolo":
            return run_yolo_detection(image_bytes, mode=mode)
        return run_detection(image_bytes, mode=mode)
    except ValueError as e:
        # If our service validates inputs and raises ValueError, surface it as a 400.
        raise HTTPException(400, str(e))
    except Exception as e:
        # Any other unexpected errors become 500. We log the full stack trace for debugging.
        logger.exception("Detection failed")
        raise HTTPException(500, f"Detection failed: {str(e)}")
