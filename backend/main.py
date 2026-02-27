"""
FastAPI backend for scene detection via SAM 3 (Segment Anything Model 3).
Exposes /detect for uploaded images.
"""
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpeg, png, webp)")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file")

    try:
        return run_detection(image_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Detection failed")
        raise HTTPException(500, f"Detection failed: {str(e)}")
