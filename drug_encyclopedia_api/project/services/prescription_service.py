# services/prescription_service.py

from rag import query_drug

def handle_prescription_query(query: str, persona: str = "patient"):
    response = query_drug(query, source="prescription", persona=persona)
    print(response)
    return response