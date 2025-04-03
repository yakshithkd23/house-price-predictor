import os
import joblib
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Load the trained model
MODEL_PATH = "models/house_price_model.pkl"

if os.path.exists(MODEL_PATH):
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data["scaler"]
else:
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

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
        # Prepare input data
        input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, 
                                total_bedrooms, population, households, median_income]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        price_prediction = model.predict(input_scaled)[0]

        return {"predicted_price": round(price_prediction, 2)}
    except Exception as e:
        return {"error": str(e)}
