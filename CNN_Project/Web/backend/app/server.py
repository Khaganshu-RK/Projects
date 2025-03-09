from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import os
import pickle

IMG_SIZE = 96

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data", "cnn_model.keras")
 
model = tf.keras.models.load_model(MODEL_PATH)

gender_labels = ["Male", "Female"]
ethnicity_labels = ["White", "Black", "Asian", "Indian", "Others"]

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = preprocess_image(image_data)

    age_pred, gender_pred, ethnicity_pred = model.predict(image)

    age = int(age_pred[0][0])
    gender = gender_labels[np.argmax(gender_pred)]
    ethnicity = ethnicity_labels[np.argmax(ethnicity_pred)]
    print(age, gender, ethnicity)

    return {"age": age, "gender": gender, "ethnicity": ethnicity}
