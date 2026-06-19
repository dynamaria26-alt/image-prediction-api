"""
Image Prediction API
---------------------
A FastAPI service that classifies images using a pretrained
MobileNetV3 model (ImageNet classes). Designed to be lightweight
enough to deploy on free/small-tier hosting (Render, Railway, Fly.io)
while still being swappable for a custom model later.
"""

import io
import logging
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-prediction-api")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# Loaded once at startup and reused across requests (avoids reloading
# weights on every call, which would be far too slow).

MODEL_STATE = {}

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_model():
    """Load a pretrained MobileNetV3-Large model and its class labels."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    return model, categories


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    model, categories = load_model()
    MODEL_STATE["model"] = model
    MODEL_STATE["categories"] = categories
    logger.info("Model loaded. Ready to serve predictions.")
    yield
    MODEL_STATE.clear()


app = FastAPI(
    title="Image Prediction API",
    description="Upload an image and get back predicted labels with confidence scores.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (adjust origins for production use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.get("/health")
def health_check():
    """Simple liveness/readiness probe for deployment platforms."""
    return {"status": "ok", "model_loaded": "model" in MODEL_STATE}


@app.get("/")
def root():
    return {
        "message": "Image Prediction API is running.",
        "docs": "/docs",
        "predict_endpoint": "/predict",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    """
    Predict the contents of an uploaded image.

    - **file**: image file (jpeg, png, webp, bmp)
    - **top_k**: number of top predictions to return (default 5, max 20)
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    top_k = max(1, min(top_k, 20))

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max size is 10MB.")

    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    model = MODEL_STATE.get("model")
    categories = MODEL_STATE.get("categories")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again shortly.")

    input_tensor = IMAGENET_TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, top_k)

    predictions = [
        {"label": categories[idx], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_indices)
    ]

    return {
        "filename": file.filename,
        "predictions": predictions,
    }
