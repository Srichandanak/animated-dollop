# routers/prescription.py

from fastapi import APIRouter
from schemas import QueryRequest
from services.prescription_service import handle_prescription_query

router = APIRouter(
    prefix="/ask/prescription",
    tags=["Prescription Drugs"]
)

@router.post("/")
def ask_prescription(request: QueryRequest):
    return handle_prescription_query(request.query)