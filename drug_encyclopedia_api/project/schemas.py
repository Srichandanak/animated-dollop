# schemas.py

from pydantic import BaseModel
from typing import Literal

class QueryRequest(BaseModel):
    query: str
    drug_name: str | None = None
    persona: Literal["patient", "student", "clinician"] = "patient"  # add this