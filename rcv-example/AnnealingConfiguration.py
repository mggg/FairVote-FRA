
from pydantic import BaseModel

class AnnealingConfiguration(BaseModel):
    """
    A model for annealing configurations.
    """
    max: float
    cycles: float
    midpoint: float
    growth: float
    cold: int
