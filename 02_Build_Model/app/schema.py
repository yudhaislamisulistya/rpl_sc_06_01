from typing import Annotated, List, Any
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    Rooms: int = Field(..., description="Number of rooms in the property")
    Bathrooms: int = Field(..., description="Number of bathrooms in the property")
    LandSize: float = Field(..., description="Size of the land in square meters")
    BuildingArea: float = Field(..., description="Building area in square meters")
    YearBuilt: int = Field(..., description="Year the property was built")
    
class PredictResponse(BaseModel):
    prediction: float
    model_version: str = "melb-v1"