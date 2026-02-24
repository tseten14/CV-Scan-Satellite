"""
FastAPI backend for entrance detection.
Exposes /detect for uploaded images.
"""
import os

if os.environ.get("TFHUB_INSECURE") or os.environ.get("PYTHONHTTPSVERIFY") == "0":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from door_detection import detect_doors


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="CV-SCAN-GEOAI Detection API",
    description="Entrance detection for facade images",
    lifespan=lifespan,
)

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
async def detect_entrances(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpeg, png, webp)")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")

    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty file")

    try:
        result = detect_doors(image_bytes)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {str(e)}")
