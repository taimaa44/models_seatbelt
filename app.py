

from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import shutil
import os
import gdown

app = FastAPI()

MODEL_PATH = "seatbelt_classifier_final.keras"

# تحميل الموديل من Google Drive
if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=14gorD0JYxYif8jiKZt-pYvdzglWyQeqH",
        MODEL_PATH,
        quiet=False
    )

# تحميل الموديل
model = tf.keras.models.load_model(MODEL_PATH)

# أسماء الكلاسات (عدليها حسبك)
class_names = ["No Seatbelt", "Seatbelt"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open("input.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread("input.jpg")
    img = cv2.resize(img, (300, 300))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        label = class_names[1]
    else:
        label = class_names[0]

    return {
        "prediction": label,
        "confidence": float(pred)
    }
