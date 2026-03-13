"""
FastAPI backend for scene detection via SAM 3 (Segment Anything Model 3).
Exposes /detect for uploaded images and /streetview for fetching street view imagery.
"""
import logging
import httpx

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from sam3_service import run_detection, load_sam3

logger = logging.getLogger("uvicorn.error")


app = FastAPI(
    title="CV-SCAN-GEOAI Detection API",
    description="Scene detection for facade / street-view images via SAM 3",
)


@app.on_event("startup")
async def startup():
    try:
        load_sam3()
    except Exception as e:
        logger.warning(f"SAM 3 preload skipped: {e}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://127.0.0.1:5173", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


_GMAPS_EMBED_KEY = "AIzaSyCmL18misQw9KdwqGaw3zHkitj8vG6QF2Y"


@app.get("/streetview-image")
async def streetview_image(
    lat: float = Query(...),
    lng: float = Query(...),
    heading: float = Query(0),
    width: int = Query(1280),
    height: int = Query(720),
):
    """Fetch a Street View image using Google's public metadata + thumbnail APIs."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
            # Resolve pano ID from coordinates via Street View metadata
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
            logger.info(f"Resolved pano_id={pano_id} for ({lat}, {lng})")

            # Fetch the actual image from the public thumbnail endpoint
            # Max size for this endpoint is ~640x640
            thumb_w = min(width, 640)
            thumb_h = min(height, 640)
            thumb_url = (
                f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail"
                f"?panoid={pano_id}"
                f"&cb_client=search.revgeo_and_hierarchicalsearch.geoname"
                f"&w={thumb_w}&h={thumb_h}"
                f"&yaw={heading}&pitch=0&thumbfov=100"
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


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    mode: str = Query("streetview", pattern="^(streetview|satellite)$"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpeg, png, webp)")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file")

    try:
        return run_detection(image_bytes, mode=mode)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Detection failed")
        raise HTTPException(500, f"Detection failed: {str(e)}")
