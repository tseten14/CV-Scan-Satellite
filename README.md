# CV-Scan-Satellite — GeoAI Suite

This repository bundles **three** tools in one dev server and API:

| Route | App | Source project |
|-------|-----|----------------|
| **`/`** | **CV-Scan** — SAM 3 entrance & building detection | Original app |
| **`/venues`** | **Venue Finder** — transit entrances on satellite maps, GTFS-derived data | `venue-finder-ai` |
| **`/spatial`** | **Spatial Visualizer** — DuckDB WASM + SQL + MapLibre (GeoJSON upload) | `spatial-sql-explorer` |

Use the **Suite** nav bar at the top to switch apps. The Python backend serves detection (`/detect`, …) and **transit search** (`/entrances`, `/entrances/cta`) used by Venue Finder.

---

## CV-Scan — Entrance & Building Detection

CV-Scan is an infrastructure mapping app. Users pick locations on an interactive map, view Google Street View imagery, and run AI-powered segmentation to detect:
- entrances in street view images
- building footprints in satellite imagery

## Features

- **Interactive map** — Leaflet with address search and spatial selection
- **Street View integration** — Street-level imagery from Google Street View
- **Detection overlay** — Polygon outlines and labels for detected objects
- **Concept prompts evaluation** — Multiple concept prompts evaluated per image

## What Does This App Do?

1. **Pick a location** — Use the interactive map to search for any address or click anywhere on the map.
2. **View the street** — The app loads a Google Street View image of that location.
3. **Run AI detection** — Choose **SAM 3** or **YOLO** in the inference bar. **SAM 3** does promptable **segmentation** (mask polygons). **YOLO** (YOLO-World when local `*world*.pt` exists, else YOLOv8 COCO) returns **bounding boxes** with similar text prompts for entrances/buildings. Satellite mode relabels YOLO boxes as `building` for exports.
4. **See results** — Each detected entrance gets a colored outline and label with a confidence score (how sure the AI is about the detection).

You can also upload your own images instead of using Street View.

## How It Works (Behind the Scenes)

The app has two parts that work together:

### Frontend (What You See)
- An interactive map powered by **Leaflet** and **OpenStreetMap**
- Address search bar that finds locations using **Nominatim** geocoding (no API key needed)
- Google Street View integration to show street-level imagery
- An overlay system that draws polygon outlines on detected objects using SVG
- A detection list showing all identified objects with their confidence scores

### Backend (The AI Brain)
- A Python server running **FastAPI** that receives images and returns detection results
- **Meta SAM 3** (default) — promptable instance segmentation (masks → polygons)
- **YOLO** (optional, `ultralytics`) — **YOLO-World** when a local `*world*.pt` is in `backend/` (needs **`openai-clip`** in venv); else **YOLOv8 COCO** if `yolov8n.pt` exists. Weights are **never auto-downloaded** by the app (avoids SSL issues). API: `POST /detect?mode=streetview|satellite&engine=sam3|yolo` — response includes `yolo_variant`: `world` \| `coco`
- **SAM 3** evaluates multiple text concepts per image (entrance-like in street view, building-like in satellite)
- Post-processing filters remove duplicates, false positives, and low-confidence detections
- Polygon outlines are extracted from the AI's segmentation masks using **OpenCV**

### Detection Pipeline
1. Image is received and resized for faster processing (street view up to 768px max dimension; satellite up to 1300px max dimension)
2. SAM 3 is run using the mode-specific concept prompts, then outputs are post-processed (NMS + filtering) to reduce duplicates and false positives
3. The model returns segmentation masks and bounding boxes for each detected object
4. Masks are converted to smooth polygon outlines
5. Post-processing removes duplicates (NMS), filters false positives, and caps detections per class
6. Results are scaled back to original image coordinates and sent to the frontend

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend Framework | React 18 + TypeScript | User interface |
| Build Tool | Vite | Fast development server and bundling |
| Map | Leaflet + React-Leaflet | Interactive map with OpenStreetMap tiles |
| Styling | Tailwind CSS + shadcn/ui | Modern dark-themed UI components |
| Backend Server | FastAPI + Uvicorn | Python API server |
| AI models | SAM 3 + YOLO (World or COCO, Ultralytics) | Segmentation vs fast boxes — pick in UI or `engine` query param |
| ML Framework | PyTorch | Runs the AI model |
| Image Processing | OpenCV + Pillow + NumPy | Mask-to-polygon conversion |
| Geocoding | Nominatim (OpenStreetMap) | Address search (no API key needed) |
| Street Imagery | Google Street View Embed | Street-level photos |

## Getting Started

### Prerequisites
- **Node.js** (v18 or later) — for the frontend
- **Python 3.10+** — for the backend
- **Hugging Face account** — to access Meta SAM 3 (it's a gated model)
  1. Create an account at [huggingface.co](https://huggingface.co)
  2. Go to [facebook/sam3](https://huggingface.co/facebook/sam3) and request access
  3. Create an access token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)

### Step 1: Install Frontend

```bash
cd frontend
npm install
```

### Step 2: Set Up Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For satellite API image fetch mode (used by `Scan Map` in satellite mode), set your HERE key:

```bash
export HERE_OIS_API_KEY=your_here_ois_key_here
```

**YOLO engine (optional):**

1. **Entrances / buildings with YOLO (recommended):** place **`yolov8s-worldv2.pt`** in `backend/` (open-vocabulary; same prompts as SAM 3). Example:

```bash
cd backend
curl -fL -o yolov8s-worldv2.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt
```

2. **Fallback only:** `yolov8n.pt` (COCO — no entrance class) if you skip World:

```bash
curl -fL -o yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

`pip install -r requirements.txt` includes **`openai-clip`** (required for YOLO-World). If HTTPS fails, copy `.pt` files from another machine. Override path with **`YOLO_WEIGHTS`**. To force COCO when both exist: **`YOLO_PREFER_COCO=1`**.

### Step 3: Run the App

Open **two terminal windows**:
Make sure you run the commands from the **project root**.

**Terminal 1 — Backend (AI server):**
```bash
export HF_TOKEN=your_huggingface_token_here
npm run backend
```
Wait until you see: `SAM 3 ready.` and `Application startup complete.`

**Terminal 2 — Frontend (web app):**
```bash
npm run dev
```

### Step 4: Open the App

Go to **http://localhost:8080** in your browser.

## How to Use

1. **Search for an address** — Type an address in the search bar on the map and click "Go"
2. **Or click the map** — Click anywhere to drop a pin
3. **View Street View or Satellite view** — Street View is used for entrances; Satellite view is used for building footprints
4. **Run detection** — Use **Scan Map** on the right (map capture / Street View fetch), or paste (Cmd+V) / **Upload image**
5. **Wait for detection** — The AI analyzes the image (timing depends on your hardware; see Performance Notes)
6. **View results** — Colored polygon outlines appear around detected objects

## What Does It Detect?

| Mode | SAM 3 labels | YOLO (engine toggle) |
|------|----------------|----------------------|
| Street View | `entrance` (prompted) | **World:** text prompts → `entrance` / vehicles; **COCO:** no door class (see UI note) |
| Satellite | `building` (prompted) | Boxes relabeled `building` for export (World or COCO, coarse) |

## Commands Reference

| Command | What It Does |
|---------|-------------|
| `npm run dev` | Start the frontend web app |
| `npm run backend` | Start the AI backend server |
| `npm run build` | Build for production deployment |
| `npm run lint` | Check code for errors |
| `npm test` | Run automated tests |

## Performance Notes

- **With GPU (CUDA/MPS):** Detection takes ~5-15 seconds per image
- **CPU only:** Detection takes ~30-90 seconds per image
- **Apple Silicon (M1/M2/M3):** Uses MPS acceleration when available
- Images are resized depending on mode: street view up to 768px max dimension, satellite up to 1300px max dimension
- Concept prompts are evaluated and results are post-processed (NMS + filtering) to remove duplicates and false positives

## Project Structure

```
CV-Scan-Satellite/
├── frontend/               # Web frontend (Vite + React + TS)
│   ├── src/                # Frontend source code
│   │   ├── components/     # React components (Map, DetectionOverlay, etc.)
│   │   ├── lib/            # Utility functions (backend detection, mock, etc.)
│   │   ├── pages/          # Page components
│   │   └── types/          # TypeScript type definitions
│   ├── public/             # Static assets
│   ├── package.json        # Frontend dependencies and scripts
│   └── vite.config.ts      # Vite configuration (dev server, API proxy)
├── backend/                # Python backend
│   ├── main.py             # FastAPI server with /detect endpoint
│   ├── sam3_service.py     # SAM 3 model loading, inference, and post-processing
│   ├── yolo_service.py     # YOLO-World (preferred) + YOLOv8 COCO fallback for engine=yolo
│   └── requirements.txt    # Python dependencies
└── package.json            # Convenience scripts (delegates to frontend/)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Backend unavailable, using mock detection" | The backend isn't running. Start it with `export HF_TOKEN=... && npm run backend` |
| Mock labels like "Main Entrance", "Side Entrance" | Same as above — these are fake labels from the fallback mock |
| "SAM 3 failed to load" | Your HF_TOKEN is missing or invalid, or you haven't been granted access to facebook/sam3 |
| YOLO errors / “model not loaded” | Add **`yolov8s-worldv2.pt`** (entrances) and/or **`yolov8n.pt`** (COCO fallback), or **`YOLO_WEIGHTS`**. Weights are not auto-downloaded. |
| `No module named 'clip'` | Run `pip install -r requirements.txt` (**`openai-clip`**) in `backend/venv`, restart the server. |
| `[SSL: CERTIFICATE_VERIFY_FAILED]` during YOLO | **YOLO-World** loads **CLIP** via HTTPS once. Fixes (pick one): (1) `pip install -r requirements.txt` includes **`certifi`** — backend sets `SSL_CERT_FILE` from it if unset. (2) Point **`SSL_CERT_FILE`** at your org’s CA bundle (corporate proxy). (3) Dev only: **`YOLO_INSECURE_SSL=1`** before `npm run backend` disables HTTPS verify for urllib in that process. (4) **`YOLO_RETRY_WITH_UNVERIFIED_SSL=1`** — retry once after SSL failure with verify off. (5) Skip CLIP: **`YOLO_PREFER_COCO=1`** and use local **`yolov8n.pt`** only (no entrances). |
| Street View **chevrons** detected as `airplane` / junk | Backend drops labels like **airplane/kite/frisbee** and small **bottom-centered** boxes (nav UI). Tune with `YOLO_STREETVIEW_UI_BOTTOM_FRAC` (default `0.14`) or `YOLO_STREETVIEW_EXTRA_DROP_LABELS=bird,boat`. |
| **YOLO-World** inaccurate / **0 detections** | Defaults: `YOLO_WORLD_STREET_CONF=0.055`, pencil tiers + **weak/side-window** cleanup (`YOLO_ENTRANCE_MIN_DISPLAY_CONF` default **0.11** drops ~6% noise). Tune `YOLO_PENCIL_*_C`, `YOLO_WEAK_SIDEWINDOW_*`. Optional: `YOLO_ENTRANCE_GEOMETRY_FILTER=1`, `YOLO_SIDELIGHT_FILTER=1`. |
| **YOLO** **0** boxes on **Scan Map** (image ~200–400px wide) | **Small map panel captures** are low-res. The app now uses **html2canvas scale ≥2** and the backend **relaxes** thresholds when `max(w,h) < ~450px` (`YOLO_TINY_*` env vars). Best: use **Street View** fetch when the pegman is active, or **upload** a full-resolution screenshot. |
| YOLO **one tiny** “entrance” **in the distance** / misses main doors | Full-res images run **`_filter_distant_micro_entrances`**: drops very small or very high-in-frame boxes unless confidence is high (`YOLO_DISTANT_MICRO_ENTRANCE_AREA_FRAC`, `YOLO_MICRO_ENTRANCE_MIN_CONF`, …). Missing main doors is still a **model** limit — use **SAM 3** when you need them. |
| Port 8000 already in use | Run `kill $(lsof -t -i :8000)` then try again |
| Very slow detection (5+ minutes) | You're running on CPU. This is expected. GPU/MPS will be much faster |
| Polygons appear outside the image | Update to the latest code — this was fixed with SVG viewBox alignment |
