from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
from tensorflow import keras

# =========================
# Paths and model loading
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_assets")

# جربي هذا أولاً إذا رفعتي الملف بصيغة .keras
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "seatbelt_classifier_final.keras")
H5_MODEL_PATH = os.path.join(MODEL_DIR, "seatbelt_model.h5")

THRESHOLD_PATH = os.path.join(MODEL_DIR, "best_threshold.npy")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.txt")

if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

# الأفضلية لملف .keras إذا كان موجودًا
if os.path.exists(KERAS_MODEL_PATH):
    MODEL_PATH = KERAS_MODEL_PATH
elif os.path.exists(H5_MODEL_PATH):
    MODEL_PATH = H5_MODEL_PATH
else:
    raise FileNotFoundError(
        f"No model file found. Expected one of: {KERAS_MODEL_PATH} or {H5_MODEL_PATH}"
    )

# تحميل الموديل
model = keras.models.load_model(MODEL_PATH, compile=False)
best_threshold = float(np.load(THRESHOLD_PATH))

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

IMG_SIZE = (300, 300)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Seatbelt Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Preprocessing
# =========================
def preprocess_pil_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# Core prediction function
# =========================
def predict_from_array(img_array):
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
        "model_file": os.path.basename(MODEL_PATH),
    }

# =========================
# FastAPI endpoints
# =========================
@app.get("/")
def root():
    return {"message": "Seatbelt API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    img_array = preprocess_image_bytes(image_bytes)
    return predict_from_array(img_array)

# =========================
# Gradio function
# =========================
def predict_gradio(image):
    if image is None:
        return {"error": "Please upload an image"}

    img_array = preprocess_pil_image(image)
    return predict_from_array(img_array)

# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# Seatbelt Detection")
    gr.Markdown("Upload an image to detect whether the seatbelt is on or off.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        json_output = gr.JSON(label="Prediction Result")

    predict_btn = gr.Button("Predict")

    predict_btn.click(
        fn=predict_gradio,
        inputs=image_input,
        outputs=json_output,
        api_name="predict"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
