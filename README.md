# CV-SCAN-GEOAI — Facade Mapper

A computer vision dashboard for urban accessibility and infrastructure mapping. Click a building on an interactive map, automatically fetch a street-level facade image, and run a detection pipeline to identify entrances and doors.

## Features

- **Interactive dark-themed map** — Leaflet with a command-center aesthetic
- **Auto-fetch street imagery** — Drops a pin and fetches a facade image automatically
- **ML-powered entrance detection** — TensorFlow object detection (EfficientDet) via Python backend
- **Split-pane layout** — Map on the left, inference pipeline on the right

## Tech Stack

- **Frontend:** React 18, TypeScript, Vite, Leaflet, Tailwind CSS, shadcn/ui
- **Backend:** FastAPI, TensorFlow, OpenCV, TensorFlow Hub (EfficientDet)

## Getting Started

### Frontend

```sh
npm install
npm run dev
```

App: `http://localhost:8080/`

### Backend (ML detection)

```sh
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

With the backend running, the frontend proxies `/api` to it. If the backend is down, it falls back to client-side COCO-SSD or mock.

### Custom model

Place `models/entrance_model.h5` in the backend folder. The server loads it at startup instead of TF Hub.

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start frontend |
| `npm run backend` | Start Python ML backend |
| `npm run build` | Production build |
| `npm run lint` | Run ESLint |
| `npm test` | Run tests |

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Production build |
| `npm run lint` | Run ESLint |
| `npm test` | Run tests |
