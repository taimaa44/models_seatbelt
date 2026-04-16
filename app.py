from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import cv2
import os
import gdown

app = FastAPI()

MODEL_PATH = "best.pt"

# تحميل الموديل من Google Drive
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=14gorD0JYxYif8jiKZt-pYvdzglWyQeqH"
    gdown.download(url, MODEL_PATH, quiet=False)

model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open("input.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model("input.jpg")

    output_path = "output.jpg"
    cv2.imwrite(output_path, results[0].plot())

    return {"message": "Prediction done"}
