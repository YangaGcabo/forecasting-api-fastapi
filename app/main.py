from fastapi import FastAPI
from app.schema import ForecastRequest, ForecastResponse
from app.model_loader import load_model
import numpy as np

app = FastAPI(title="Forecasting API")

model = load_model()

@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    return ForecastResponse(prediction=float(prediction))
