from pydantic import BaseModel
from typing import List

class ForecastRequest(BaseModel):
    features: List[float]

class ForecastResponse(BaseModel):
    prediction: float
