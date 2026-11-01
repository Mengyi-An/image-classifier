"""
Image classification API.

Wraps the trained model in an HTTP service: upload an image and get its predicted class.

Endpoints:
    GET  /health               check that the service is ready
    GET  /classes              list the 10 classes
    GET  /classes/{name}       describe one class
    POST /predict              classify one image
    POST /predict/batch        classify several images at once
"""

import io
import os
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import IMAGE_SIZE, NORM_MEAN, NORM_STD, ImageClassifier

# The trained model checkpoint: data/model.pt
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Uploads are read fully into memory, so cap their size (10 MB per file, at most 20 files
# per batch) to keep one huge request from taking the service down
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 20

model = None
model_metadata = {}

TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # the model was trained on 32x32 images
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

CLASS_DESCRIPTIONS = {
    "airplane": "Fixed-wing aircraft",
    "automobile": "Car or similar road vehicle",
    "bird": "Any species of bird",
    "cat": "Domestic cat",
    "deer": "Deer or similar deer-like animal",
    "dog": "Domestic dog",
    "frog": "Frog or toad",
    "horse": "Horse or similar equine",
    "ship": "Large watercraft",
    "truck": "Large road vehicle for cargo",
}


def load_model():
    global model, model_metadata
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Run src/train.py first.")

    # map_location lets a model trained on a GPU load on a CPU-only machine
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    # train.py writes a bundle (weights + params + seed + metrics). Older
    # checkpoints are a bare state_dict, so both are accepted.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_metadata = {
            "params": checkpoint.get("params", {}),
            "seed": checkpoint.get("seed"),
            "metrics": checkpoint.get("metrics", {}),
        }
    else:
        state_dict = checkpoint
        model_metadata = {}

    m = ImageClassifier()
    m.load_state_dict(state_dict)
    m.eval()
    model = m
    print(f"Model loaded on {DEVICE}. Metadata: {model_metadata or 'none (bare state_dict)'}")


def read_upload(file: UploadFile) -> bytes:
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    return contents


# Decode the bytes into an image and return the top-k predictions.
# convert("RGB") handles grayscale images and PNGs with an alpha channel.
# Anything that cannot be decoded is the client's fault, so it returns 400.
def classify_image(image_bytes: bytes, top_k: int, min_confidence: float) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    # The k most likely classes and their probabilities
    top_prob, top_idx = probs.topk(top_k)
    top_predictions = [
        {"class": ImageClassifier.CLASSES[i], "probability": round(p, 4)}
        for i, p in zip(top_idx.tolist(), top_prob.tolist())
    ]

    top_class = top_predictions[0]["class"]
    top_confidence = top_predictions[0]["probability"]
    confident = top_confidence >= min_confidence

    # Below the requested confidence, report "uncertain" instead of a class
    return {
        "prediction": top_class if confident else "uncertain",
        "confidence": top_confidence,
        "confident": confident,
        "top_k": top_predictions,
    }


# Startup hook: the code before `yield` runs once when the service starts, so the model is
# loaded a single time and shared by all requests instead of being re-read per request.
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Image Classifier API",
    description="Classifies images into CIFAR-10 categories using a custom CNN.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Image Classifier API is running. Visit /docs for usage."}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok", "device": DEVICE}


@app.get("/classes")
def get_classes():
    return {"classes": ImageClassifier.CLASSES}


# {name} is a path parameter: /classes/cat calls this with name="cat"
@app.get("/classes/{name}")
def get_class_info(name: str):
    if name not in CLASS_DESCRIPTIONS:
        raise HTTPException(status_code=404, detail=f"Class '{name}' not found.")
    return {"class": name, "description": CLASS_DESCRIPTIONS[name]}


# FastAPI validates the parameters from their declarations: the file is required, and
# top_k / min_confidence outside their ranges get an automatic 422 response.
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=3, ge=1, le=10, description="Number of top predictions to return"),
    min_confidence: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence to return a prediction"
    ),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    contents = read_upload(file)
    return classify_image(contents, top_k, min_confidence)


@app.post("/predict/batch")
def predict_batch(
    files: list[UploadFile] = File(...),
    top_k: int = Query(default=3, ge=1, le=10),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"Batch of {len(files)} exceeds the limit of {MAX_BATCH_FILES} files.",
        )

    results = []
    for file in files:
        contents = read_upload(file)
        result = classify_image(contents, top_k, min_confidence)
        results.append({"filename": file.filename, **result})

    return {
        "total": len(results),
        "results": results,
    }
