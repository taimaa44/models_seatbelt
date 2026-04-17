from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model_assets")

MODEL_PATH = os.path.join(MODEL_DIR, "seatbelt_classifier_final.keras")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "best_threshold.npy")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.txt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
best_threshold = float(np.load(THRESHOLD_PATH))

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

IMG_SIZE = (300, 300)

app = FastAPI(title="Seatbelt Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # لاحقًا بدليها بدومين موقعك
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def root():
    return {"message": "Seatbelt API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    img_array = preprocess_image(image_bytes)

    prob = float(model.predict(img_array, verbose=0)[0][0])
    pred_idx = 1 if prob >= best_threshold else 0
    pred_class = class_names[pred_idx]
    confidence = prob if pred_idx == 1 else (1.0 - prob)

    return {
        "predicted_class": pred_class,
        "seatbelt_on": bool(pred_idx == 1),
        "confidence": round(float(confidence), 6),
        "raw_probability": round(float(prob), 6),
        "threshold": round(float(best_threshold), 6),
    }
