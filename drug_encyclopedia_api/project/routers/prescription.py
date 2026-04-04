# routers/prescription.py

from fastapi import APIRouter
from schemas import QueryRequest
from services.prescription_service import handle_prescription_query
from rag import clinical_query  # ← just this, remove query_with_persona entirely

router = APIRouter(
    prefix="/ask/prescription",
    tags=["Prescription Drugs"]
)

@router.get("/clinical")
def clinical(q: str):
    return clinical_query(q)

@router.post("/")
def ask_prescription(request: QueryRequest):
    return handle_prescription_query(request.query, request.persona)
