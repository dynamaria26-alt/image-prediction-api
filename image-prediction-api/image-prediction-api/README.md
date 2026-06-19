# Image Prediction API

A simple, deployable image classification API built with **FastAPI** and a
pretrained **MobileNetV3-Large** model (1000 ImageNet classes). Upload an
image, get back the top predicted labels with confidence scores.

## Endpoints

| Method | Path       | Description                                  |
|--------|------------|-----------------------------------------------|
| GET    | `/`        | Basic info                                    |
| GET    | `/health`  | Health check (used by deploy platforms)       |
| POST   | `/predict` | Upload an image, get predictions              |
| GET    | `/docs`    | Interactive Swagger UI (auto-generated)       |

### `POST /predict`

**Form-data body:**
- `file` (required): image file — jpeg, png, webp, or bmp, max 10MB
- `top_k` (optional, query param): number of predictions to return (default 5, max 20)

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict?top_k=3" \
  -F "file=@/path/to/your/image.jpg"
```

**Example response:**
```json
{
  "filename": "image.jpg",
  "predictions": [
    { "label": "golden retriever", "confidence": 0.8421 },
    { "label": "Labrador retriever", "confidence": 0.0931 },
    { "label": "kuvasz", "confidence": 0.0204 }
  ]
}
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` to try it interactively.

## Run with Docker

```bash
docker build -t image-prediction-api .
docker run -p 8000:8000 image-prediction-api
```

## Deploy

This repo works out of the box on any platform that supports Docker or a
Python buildpack:

- **Render**: New → Web Service → connect repo → it will detect the
  Dockerfile automatically. No extra config needed.
- **Railway**: New Project → Deploy from GitHub repo → detects Dockerfile.
- **Fly.io**: `fly launch` from the repo root (it will detect the Dockerfile).
- **Any VPS**: `docker build` + `docker run` as shown above, behind nginx/Caddy.

All these platforms inject a `PORT` environment variable automatically —
the Dockerfile's `CMD` already respects `$PORT`.

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: image prediction API"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Swapping in your own model

Replace the `load_model()` function in `app/main.py` with logic to load
your own PyTorch/ONNX/TensorFlow model and update the preprocessing
transform + label list to match. Everything else (endpoints, validation,
error handling) stays the same.

## Notes

- First request after startup may be a bit slow while PyTorch initializes;
  subsequent requests are fast.
- CORS is currently open (`*`) for ease of testing — restrict
  `allow_origins` in `app/main.py` before going to production.
