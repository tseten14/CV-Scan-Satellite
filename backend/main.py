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


@app.get("/streetview-image")
async def streetview_image(
    lat: float = Query(...),
    lng: float = Query(...),
    heading: float = Query(0),
    width: int = Query(640),
    height: int = Query(480),
):
    """Fetch a Street View image by first resolving the panorama ID, then stitching tiles."""
    import io
    from PIL import Image as PILImage

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        async with httpx.AsyncClient(timeout=15, headers=headers, follow_redirects=True) as client:
            # Step 1: resolve pano ID from coordinates via metadata endpoint
            meta_url = (
                f"https://maps.googleapis.com/maps/api/js/GeoPhotoService.GetMetadata"
                f"?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0"
                f"!2m2!1d{lng}!2d{lat}"
                f"!3m3!1m2!1e2!2e1!4m6!1e1!1e2!1e3!1e4!1e8!1e6"
            )
            meta_resp = await client.get(meta_url)

            pano_id = None
            if meta_resp.status_code == 200:
                import re
                # Extract pano ID from the response (it's in a nested array format)
                match = re.search(r'\[\[2,"([A-Za-z0-9_\-]+)"', meta_resp.text)
                if match:
                    pano_id = match.group(1)

            if not pano_id:
                raise HTTPException(404, "No street view panorama found at this location")

            # Step 2: fetch a single high-res tile (zoom 2 gives ~1664x832 per tile)
            tile_url = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom=2&x=0&y=0"
            tile_resp = await client.get(tile_url)
            if tile_resp.status_code != 200 or b"image" not in (tile_resp.headers.get("content-type", "").encode()):
                # Fallback: try zoom 1
                tile_url = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom=1&x=0&y=0"
                tile_resp = await client.get(tile_url)

            if tile_resp.status_code != 200:
                raise HTTPException(502, "Failed to fetch street view tiles")

            # Stitch multiple tiles for a wider view at zoom 2 (4 columns x 2 rows)
            zoom = 2
            cols, rows = 4, 2
            tile_w, tile_h = 512, 512
            pano = PILImage.new("RGB", (tile_w * cols, tile_h * rows))
            for ty in range(rows):
                for tx in range(cols):
                    t_url = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={tx}&y={ty}"
                    t_resp = await client.get(t_url)
                    if t_resp.status_code == 200:
                        tile_img = PILImage.open(io.BytesIO(t_resp.content))
                        pano.paste(tile_img, (tx * tile_w, ty * tile_h))

            # Crop to a front-facing view (center portion)
            full_w, full_h = pano.size
            # Take center 1/4 width for ~90 degree FOV
            crop_w = full_w // 4
            left = (full_w - crop_w) // 2
            cropped = pano.crop((left, 0, left + crop_w, full_h))
            cropped = cropped.resize((width, height), PILImage.Resampling.LANCZOS)

            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            return Response(content=buf.read(), media_type="image/jpeg")

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
