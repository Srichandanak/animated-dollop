# routers/drug_resolver.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from rag import DRUG_MAP

router = APIRouter(prefix="/drugs", tags=["Drug Resolution"])

class ResolveRequest(BaseModel):
    query: str

@router.post("/resolve")
def resolve_drug(req: ResolveRequest):
    query_lower = req.query.lower().strip()
    brand_map = DRUG_MAP["brand_to_generic"]
    generics = DRUG_MAP["generics"]

    # direct generic match
    if query_lower in generics:
        return {"resolved": [generics[query_lower]], "original": req.query}

    # brand name match
    if query_lower in brand_map:
        generic_key = brand_map[query_lower]
        return {"resolved": [generics[generic_key]], "original": req.query}

    # nothing found
    raise HTTPException(status_code=404, detail="Drug not found")

@router.get("/list")
def list_drugs():
    drugs = [
        {"key": k, "display_name": v["display_name"]}
        for k, v in DRUG_MAP["generics"].items()
    ]
    return {"drugs": drugs}