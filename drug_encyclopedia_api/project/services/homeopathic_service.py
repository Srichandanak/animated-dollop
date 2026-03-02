# services/homeopathic_service.py

from rag import query_drug

def handle_homeopathic_query(query: str):
    # You can add homeopathic-specific logic here
    response = query_drug(query, source="homeopathic")
    return response