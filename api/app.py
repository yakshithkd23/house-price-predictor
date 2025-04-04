import os
import joblib
import gdown
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Define model path
MODEL_PATH = "models/house_price_model.pkl"

# Google Drive File ID (Extracted from the link)
DRIVE_FILE_ID = "18gspJPwI--8na-mJnKEe6L5-zmmqrrp0"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Download the model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Load the trained model
if os.path.exists(MODEL_PATH):
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data["scaler"]
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"❌ Model file {MODEL_PATH} not found!")

# Initialize FastAPI
app = FastAPI()

# Mount static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_price(
    longitude: float = Form(...),
    latitude: float = Form(...),
    housing_median_age: float = Form(...),
    total_rooms: float = Form(...),
    total_bedrooms: float = Form(...),
    population: float = Form(...),
    households: float = Form(...),
    median_income: float = Form(...)
):
    try:
        input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, 
                                total_bedrooms, population, households, median_income]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        price_prediction = model.predict(input_scaled)[0]

        return {"predicted_price": round(price_prediction, 2)}
    except Exception as e:
        return {"error": str(e)}
