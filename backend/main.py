# FastAPI backend for scene detection: SAM 3, YOLO v9 (custom doors), and YOLO v8-world.
# Exposes /detect for uploaded images and /streetview for fetching street view imagery.
import logging
import httpx

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from sam3_service import run_detection, load_sam3
from yolo_service import run_yolo_detection
from yolo_world_service import run_yolo_world_detection
from entrances import get_entrances, get_cta_entrances

logger = logging.getLogger("uvicorn.error")


app = FastAPI(
    title="CV-SCAN-GEOAI Detection API",
    description="Scene detection via SAM 3, YOLO v9 (door weights), or YOLO v8-world — ?engine=sam3|yolo|yolo_world",
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
)


@app.get("/health")
async def health():
    # Simple health check endpoint for debugging and monitoring.

    # This is intentionally lightweight and does not require the SAM 3 model.
    return {"status": "ok"}


_GMAPS_EMBED_KEY = "AIzaSyCmL18misQw9KdwqGaw3zHkitj8vG6QF2Y"


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
    engine: str = Query(
        "sam3",
        pattern="^(sam3|yolo|yolov9|yolo26|yolo_world|yoloworld)$",
    ),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpeg, png, webp)")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file")

    # Normalize legacy query values to the single custom-YOLO stack (YOLO v9 weights).
    yolo_aliases = ("yolo", "yolov9", "yolo26")
    yolo_world_aliases = ("yolo_world", "yoloworld")
    if engine in yolo_aliases:
        engine_effective = "yolo"
    elif engine in yolo_world_aliases:
        engine_effective = "yolo_world"
    else:
        engine_effective = engine

    logger.info("POST /detect mode=%s engine=%s bytes=%s", mode, engine_effective, len(image_bytes))

    try:
        if engine_effective == "yolo":
            return run_yolo_detection(image_bytes, mode=mode)
        if engine_effective == "yolo_world":
            return run_yolo_world_detection(image_bytes, mode=mode)
        return run_detection(image_bytes, mode=mode)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Detection failed")
        raise HTTPException(500, f"Detection failed: {str(e)}")
