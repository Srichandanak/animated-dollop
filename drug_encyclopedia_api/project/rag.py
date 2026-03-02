# rag.py

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# -------------------------
# LOAD DATA (MODULE LEVEL)
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RX_PATH = os.path.join(BASE_DIR, "prescription_structured.csv")
HOMEO_PATH = os.path.join(BASE_DIR, "homeopathic_structured.csv")

rx_df = pd.read_csv(RX_PATH).fillna("")
homeo_df = pd.read_csv(HOMEO_PATH).fillna("")

# -------------------------
# LOAD MODEL (ONCE)
# -------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# INIT PERSISTENT CHROMA
# -------------------------

chroma_path = os.path.join(BASE_DIR, "chroma_store")

client = chromadb.Client(
    Settings(
        persist_directory=chroma_path,
        is_persistent=True
    )
)

collection = client.get_or_create_collection("drug_collection")


# -------------------------
# INDEX DATA (ONLY IF EMPTY)
# -------------------------

def index_data():

    if collection.count() > 0:
        return  # Already indexed

    documents = []
    metadatas = []
    ids = []

    counter = 0

    # Prescription drugs
    for _, row in rx_df.iterrows():

        combined = f"""
        Drug: {row['drug']}
        Uses: {row['uses']}
        Dosage: {row['dosage']}
        Side Effects: {row.get('side_effects', '')}
        Warnings: {row.get('warnings', '')}
        Interactions: {row.get('interactions', '')}
        Contraindications: {row.get('contraindications', '')}
        """

        documents.append(combined)
        metadatas.append({
            "drug": row["drug"],
            "type": "prescription"
        })
        ids.append(f"rx_{counter}")
        counter += 1

    # Homeopathic drugs
    for _, row in homeo_df.iterrows():

        combined = f"""
        Drug: {row['drug']}
        Uses: {row['uses']}
        Dosage: {row['dosage']}
        Warnings: {row.get('warnings', '')}
        """

        documents.append(combined)
        metadatas.append({
            "drug": row["drug"],
            "type": "homeopathic"
        })
        ids.append(f"homeo_{counter}")
        counter += 1

    embeddings = model.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    # client.persist()


# Call once on import
# index_data()


# -------------------------
# QUERY FUNCTION
# -------------------------

def query_drug(query: str, source: str):

    query_lower = query.lower()

    if source == "prescription":
        df = rx_df
    elif source == "homeopathic":
        df = homeo_df
    else:
        return {"message": "Invalid source"}

    # 1️⃣ Exact match first (fast and accurate)
    for _, row in df.iterrows():
        if row["drug"].lower() in query_lower:

            response = {
                "drug": row["drug"],
                "uses": row["uses"],
                "dosage": row["dosage"],
                "warnings": row.get("warnings", "")
            }

            if source == "prescription":
                response["side_effects"] = row.get("side_effects", "")
                response["interactions"] = row.get("interactions", "")
                response["contraindications"] = row.get("contraindications", "")
            else:
                response["side_effects"] = ""
                response["interactions"] = ""
                response["contraindications"] = ""

            return response

    # 2️⃣ Semantic fallback
    results = collection.query(
        query_texts=[query],
        n_results=1
    )

    if not results["documents"] or not results["documents"][0]:
        return {"message": "No drug found."}

    metadata = results["metadatas"][0][0]
    drug_name = metadata["drug"]

    row = df[df["drug"] == drug_name]

    if row.empty:
        return {"message": "No drug found."}

    row = row.iloc[0]

    response = {
        "drug": row["drug"],
        "uses": row["uses"],
        "dosage": row["dosage"],
        "warnings": row.get("warnings", "")
    }

    if source == "prescription":
        response["side_effects"] = row.get("side_effects", "")
        response["interactions"] = row.get("interactions", "")
        response["contraindications"] = row.get("contraindications", "")
    else:
        response["side_effects"] = ""
        response["interactions"] = ""
        response["contraindications"] = ""

    return response