# services/homeopathic_service.py

from rag import query_drug

def handle_homeopathic_query(query: str, persona: str = "patient"):
    response = query_drug(query, source="homeopathic", persona=persona)
    return response