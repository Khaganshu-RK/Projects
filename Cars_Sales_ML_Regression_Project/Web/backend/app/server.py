from fastapi import FastAPI
import pickle
import yaml
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np

app = FastAPI()

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "data", "input_preprocessor.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model.pkl")
Y_ENCODER_PATH = os.path.join(BASE_DIR, "data", "target_encoder.pkl")
YAML_PATH = os.path.join(BASE_DIR, "data", "data_schema.yaml")

# Load pre-trained objects
try:
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("One or more pickle files not found.")

# Load schema
with open(YAML_PATH, "r") as f:
    columns = yaml.safe_load(f)

# Load target encoder
if os.path.exists(Y_ENCODER_PATH):
    with open(Y_ENCODER_PATH, "rb") as f:
        target_encoder = pickle.load(f)
else:
    target_encoder = None

# Define request model
class InputData(BaseModel):
    data: dict

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/predict")
def predict(input_data: InputData):
    """Handles predictions with preprocessing and target decoding."""
    
    user_input = input_data.data
    print("User Input:", user_input)

    # Prepare input array
    input_array = []
    col_name = []
    
    # Process categorical features
    for key in columns["categorical"]:
        input_array.append(user_input[key])
        col_name.append(key)

    # Process numerical features
    for key in columns["numerical"]:
        if key != "sellingprice":  # As 'sellingprice' is the target variable
            if columns["numerical"][key]["type"] == 'int':
                input_array.append(int(user_input[key]))
            elif columns["numerical"][key]["type"] == 'float':
                input_array.append(float(user_input[key]))
            col_name.append(key)

    # Convert input to DataFrame
    input_array = pd.DataFrame([input_array], columns=col_name)

    print("Input Data:", input_array)

    print("Expected Columns:", preprocessor.feature_names_in_)
    print("Input Data Shape:", input_array.shape)

    # Validate input shape
    if input_array.shape[1] != preprocessor.feature_names_in_.shape[0]:
        raise ValueError("Input data does not match the expected columns.")

    # Preprocess input
    processed_data = preprocessor.transform(input_array)
    print("Processed Data Shape:", processed_data.shape)

    # Make predictions
    prediction = model.predict(processed_data)

    # Ensure predictions have the right shape
    print("Raw Prediction:", prediction)

    if target_encoder is not None:
        if isinstance(target_encoder, np.ndarray):
            ## If encoder is OneHotEncoder, reverse transform using argmax
            # prediction = prediction.argmax(axis=1)
            # prediction = target_encoder.inverse_transform(prediction.reshape(-1, 1))
            prediction = target_encoder.predict(processed_data)
        else:
            ## LabelEncoder case (binary classification)
            prediction = prediction.astype(int)
            prediction = target_encoder.inverse_transform(prediction)

    print("Final Prediction:", prediction.tolist())
    
    return {"prediction": prediction.tolist()}