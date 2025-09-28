from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    recency_days: float
    orders: int
    monetary: float
    tenure_days: float
    avg_discount: float
    return_rate: float
    category_diversity: int
