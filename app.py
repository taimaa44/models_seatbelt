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

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ model loaded")

class_names = ["No Seatbelt", "Seatbelt"]

@app.get("/")
def home():
    return {"message": "API is working 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open("input.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread("input.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 🔥 مهم
    img = cv2.resize(img, (300, 300))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    label = class_names[1] if pred > 0.5 else class_names[0]

    return {
        "prediction": label,
        "confidence": float(pred)
    }

# تشغيل السيرفر
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
