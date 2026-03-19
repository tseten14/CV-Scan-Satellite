# CV-Scan-Satellite — Entrance & Building Detection

CV-Scan-Satellite is an infrastructure mapping app. Users pick locations on an interactive map, view Google Street View imagery, and run AI-powered segmentation to detect:
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
3. **Run AI detection** — The image is sent to Meta's **SAM 3 (Segment Anything Model 3)**, which detects entrances in the scene and draws precise polygon outlines around each one.
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
- **Meta SAM 3** (Segment Anything Model 3) — a state-of-the-art AI model that can segment any object in an image given a text description
- The model evaluates multiple concept prompts per image (e.g., entrance-like concepts in street view mode, building-like concepts in satellite mode)
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
| AI Model | Meta SAM 3 (via Hugging Face Transformers) | Object detection and segmentation |
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
4. **Screenshot / Capture** — Take a screenshot of the current view (Cmd+Shift+4 on Mac, or use the upload button)
5. **Paste or upload** — Paste the screenshot (Cmd+V) or click "Upload Image" in the detection panel
6. **Wait for detection** — The AI analyzes the image (typically 15-60 seconds depending on your hardware)
7. **View results** — Colored polygon outlines appear around detected objects

## What Does It Detect?

| Mode | Output labels |
|------|---------------|
| Street View | `entrance` |
| Satellite | `building` |

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
│   └── requirements.txt    # Python dependencies
└── package.json            # Convenience scripts (delegates to frontend/)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Backend unavailable, using mock detection" | The backend isn't running. Start it with `export HF_TOKEN=... && npm run backend` |
| Mock labels like "Main Entrance", "Side Entrance" | Same as above — these are fake labels from the fallback mock |
| "SAM 3 failed to load" | Your HF_TOKEN is missing or invalid, or you haven't been granted access to facebook/sam3 |
| Port 8000 already in use | Run `kill $(lsof -t -i :8000)` then try again |
| Very slow detection (5+ minutes) | You're running on CPU. This is expected. GPU/MPS will be much faster |
| Polygons appear outside the image | Update to the latest code — this was fixed with SVG viewBox alignment |
