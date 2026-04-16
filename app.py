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
    print("⬇️ Downloading model...")
    gdown.download(
        "https://drive.google.com/uc?id=14gorD0JYxYif8jiKZt-pYvdzglWyQeqH",
        MODEL_PATH,
        quiet=False
    )

model = None

class_names = ["No Seatbelt", "Seatbelt"]

@app.get("/")
def home():
    return {"message": "API is working 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model

    try:
        # تحميل الموديل أول مرة فقط
        if model is None:
            print("⏳ Loading model...")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded")

        # حفظ الصورة
        file_path = "input.jpg"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # قراءة الصورة
        img = cv2.imread(file_path)

        if img is None:
            return {"error": "Image not read correctly"}

        # معالجة الصورة
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # prediction
        pred = model.predict(img, verbose=0)[0][0]

        label = class_names[1] if pred > 0.5 else class_names[0]

        return {
            "prediction": label,
            "confidence": float(pred)
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {
            "error": str(e)
        }

# تشغيل السيرفر
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
