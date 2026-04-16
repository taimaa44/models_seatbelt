from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import shutil
import os
import gdown

app = FastAPI()

MODEL_PATH = "seatbelt_classifier_final.keras"

# تحميل الموديل
if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=14gorD0JYxYif8jiKZt-pYvdzglWyQeqH",
        MODEL_PATH,
        quiet=False
    )

# تحميل الموديل (بدون compile)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

class_names = ["No Seatbelt", "Seatbelt"]

@app.get("/")
def home():
    return {"message": "API is working 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open("input.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread("input.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    label = class_names[1] if pred >= 0.5 else class_names[0]

    confidence = float(pred) if pred >= 0.5 else float(1 - pred)

    return {
        "prediction": label,
        "confidence": confidence
    }
