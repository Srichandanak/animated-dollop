# services/prescription_service.py

from rag import query_drug

def handle_prescription_query(query: str):
    # You can add prescription-specific logic here
    response = query_drug(query, source="prescription")
    print(response)
    return response