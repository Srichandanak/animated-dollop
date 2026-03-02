# routers/homeopathic.py

from fastapi import APIRouter
from schemas import QueryRequest
from services.homeopathic_service import handle_homeopathic_query

router = APIRouter(
    prefix="/ask/homeopathic",
    tags=["Homeopathic Drugs"]
)

@router.post("/")
def ask_homeopathic(request: QueryRequest):
    return handle_homeopathic_query(request.query)