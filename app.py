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
    url = "https://drive.google.com/uc?id=14gorD0JYxYif8jiKZt-pYvdzglWyQeqH"
    gdown.download(url, MODEL_PATH, quiet=False)

# تحميل الموديل
model = tf.keras.models.load_model(MODEL_PATH)

# prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open("input.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # قراءة الصورة
    img = cv2.imread("input.jpg")
    img = cv2.resize(img, (224, 224))  # حسب تدريبك
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction
    pred = model.predict(img)
    result = float(pred[0][0])

    # مثال classification
    if result > 0.5:
        label = "Seatbelt"
    else:
        label = "No Seatbelt"

    return {
        "prediction": label,
        "confidence": result
    }
