
# # rag.py

# import os
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.config import Settings
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os

# load_dotenv()
# # -------------------------
# # LOAD DATA
# # -------------------------

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# RX_PATH = os.path.join(BASE_DIR, "prescription_structured.csv")
# HOMEO_PATH = os.path.join(BASE_DIR, "homeopathic_structured.csv")

# rx_df = pd.read_csv(RX_PATH).fillna("")
# homeo_df = pd.read_csv(HOMEO_PATH).fillna("")

# # -------------------------
# # LOAD EMBEDDING MODEL
# # -------------------------

# model = SentenceTransformer("all-MiniLM-L6-v2")

# # -------------------------
# # INIT CHROMA
# # -------------------------

# chroma_path = os.path.join(BASE_DIR, "chroma_store")

# client = chromadb.Client(
#     Settings(
#         persist_directory=chroma_path,
#         is_persistent=True
#     )
# )

# collection = client.get_or_create_collection("drug_collection")

# # -------------------------
# # INDEX DATA (UNCHANGED)
# # -------------------------

# def index_data():

#     if collection.count() > 0:
#         return

#     documents = []
#     metadatas = []
#     ids = []
#     counter = 0

#     for _, row in rx_df.iterrows():
#         combined = f"""
#         Drug: {row['drug']}
#         Uses: {row['uses']}
#         Dosage: {row['dosage']}
#         Side Effects: {row.get('side_effects', '')}
#         Warnings: {row.get('warnings', '')}
#         Interactions: {row.get('interactions', '')}
#         Contraindications: {row.get('contraindications', '')}
#         """

#         documents.append(combined)
#         metadatas.append({
#             "drug": row["drug"],
#             "type": "prescription"
#         })
#         ids.append(f"rx_{counter}")
#         counter += 1

#     for _, row in homeo_df.iterrows():
#         combined = f"""
#         Drug: {row['drug']}
#         Uses: {row['uses']}
#         Dosage: {row['dosage']}
#         Warnings: {row.get('warnings', '')}
#         """

#         documents.append(combined)
#         metadatas.append({
#             "drug": row["drug"],
#             "type": "homeopathic"
#         })
#         ids.append(f"homeo_{counter}")
#         counter += 1

#     embeddings = model.encode(documents).tolist()

#     collection.add(
#         documents=documents,
#         embeddings=embeddings,
#         metadatas=metadatas,
#         ids=ids
#     )

# # -------------------------
# # ORIGINAL QUERY FUNCTION (UNCHANGED)
# # -------------------------

# def query_drug(query: str, source: str):

#     query_lower = query.lower()

#     if source == "prescription":
#         df = rx_df
#     elif source == "homeopathic":
#         df = homeo_df
#     else:
#         return {"message": "Invalid source"}

#     for _, row in df.iterrows():
#         if row["drug"].lower() in query_lower:

#             response = {
#                 "drug": row["drug"],
#                 "uses": row["uses"],
#                 "dosage": row["dosage"],
#                 "warnings": row.get("warnings", "")
#             }

#             if source == "prescription":
#                 response["side_effects"] = row.get("side_effects", "")
#                 response["interactions"] = row.get("interactions", "")
#                 response["contraindications"] = row.get("contraindications", "")
#             else:
#                 response["side_effects"] = ""
#                 response["interactions"] = ""
#                 response["contraindications"] = ""

#             return response

#     # semantic fallback
#     query_embedding = model.encode([query]).tolist()

#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=1
#     )

#     if not results["documents"] or not results["documents"][0]:
#         return {"message": "No drug found."}

#     metadata = results["metadatas"][0][0]
#     drug_name = metadata["drug"]

#     row = df[df["drug"] == drug_name]

#     if row.empty:
#         return {"message": "No drug found."}

#     row = row.iloc[0]

#     response = {
#         "drug": row["drug"],
#         "uses": row["uses"],
#         "dosage": row["dosage"],
#         "warnings": row.get("warnings", "")
#     }

#     if source == "prescription":
#         response["side_effects"] = row.get("side_effects", "")
#         response["interactions"] = row.get("interactions", "")
#         response["contraindications"] = row.get("contraindications", "")
#     else:
#         response["side_effects"] = ""
#         response["interactions"] = ""
#         response["contraindications"] = ""

#     return response


# # ==========================================================
# # NEW: GROQ CLINICAL REASONING LAYER
# # ==========================================================

# import os
# from langchain_groq import ChatGroq


# def get_llm():
#     return ChatGroq(
#         model_name="llama-3.1-8b-instant",
#         temperature=0,
#         max_tokens=1024,  # prevents runaway outputs
#         groq_api_key=os.getenv("GROQ_API_KEY")
#     )

# def clinical_query(question: str):

#     # Retrieve top 3 relevant drug documents
#     query_embedding = model.encode([question]).tolist()

#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=1
#     )

#     if not results["documents"] or not results["documents"][0]:
#         return {"message": "No relevant drug information found."}

#     context = "\n\n".join(results["documents"][0])

#     llm = get_llm()

#     prompt = f"""
# You are a clinical drug safety assistant.

# Use ONLY the provided drug label context.
# If answer is not present, say:
# "Not mentioned in the provided label."

# Context:
# {context}

# Question:
# {question}

# Respond strictly in this format:

# Safety Assessment:
# Reason:
# Source Evidence:
# """

#     response = llm.invoke(prompt)

#     return {
#         "question": question,
#         "analysis": response.content
#     }

# rag.py

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
# rag.py — add this after load_dotenv()

import json
from pathlib import Path
BASE_DIR = "D:/animated-dollop/drug_encyclopedia_api/project"
# Load drug map for brand → generic resolution
_drug_map_path = os.path.join(BASE_DIR, "drug_map.json")
if os.path.exists(_drug_map_path):
    DRUG_MAP = json.loads(Path(_drug_map_path).read_text())
else:
    # fallback empty structure if file doesn't exist yet
    DRUG_MAP = {"brand_to_generic": {}, "generics": {}}
# -------------------------
# LOAD DATA (UNCHANGED)
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RX_PATH = os.path.join(BASE_DIR, "prescription_structured.csv")
HOMEO_PATH = os.path.join(BASE_DIR, "homeopathic_structured.csv")

rx_df = pd.read_csv(RX_PATH).fillna("")
homeo_df = pd.read_csv(HOMEO_PATH).fillna("")

# -------------------------
# LOAD EMBEDDING MODEL (UNCHANGED)
# -------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# INIT CHROMA (UNCHANGED)
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
# INDEX DATA (UNCHANGED)
# -------------------------

def index_data():

    if collection.count() > 0:
        return

    documents = []
    metadatas = []
    ids = []
    counter = 0

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
        metadatas.append({"drug": row["drug"], "type": "prescription"})
        ids.append(f"rx_{counter}")
        counter += 1

    for _, row in homeo_df.iterrows():
        combined = f"""
        Drug: {row['drug']}
        Uses: {row['uses']}
        Dosage: {row['dosage']}
        Warnings: {row.get('warnings', '')}
        """
        documents.append(combined)
        metadatas.append({"drug": row["drug"], "type": "homeopathic"})
        ids.append(f"homeo_{counter}")
        counter += 1

    embeddings = model.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

# -------------------------
# ORIGINAL QUERY FUNCTION (UNCHANGED)
# -------------------------

def query_drug(query: str, source: str, persona: str = "patient"):
    # NOTE: persona is accepted here and passed to the
    # persona layer AFTER the drug data is retrieved.
    # The retrieval logic below is completely unchanged.

    query_lower = query.lower()

    if source == "prescription":
        df = rx_df
    elif source == "homeopathic":
        df = homeo_df
    else:
        return {"message": "Invalid source"}

    # --- exact match first (unchanged) ---
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

            # NEW: once we have the raw drug data, run it through
            # the persona layer before returning
            return apply_persona(drug_data=response, question=query, persona=persona)

    # --- semantic fallback (unchanged) ---
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
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

    # NEW: same here — apply persona before returning semantic fallback result
    return apply_persona(drug_data=response, question=query, persona=persona)


# ==========================================================
# NEW: PERSONA SYSTEM PROMPTS
# One prompt per persona. These control how Groq rewrites
# the raw drug data retrieved above.
# ==========================================================

PERSONA_PROMPTS = {
    "patient": """You are a friendly health assistant explaining medication to a patient with no medical background.
Use only plain English. No Latin, no abbreviations, no jargon.
Format your response as:
- One simple sentence saying what the drug is for
- 3 to 5 bullet points covering key information
- One line starting with "Ask your doctor if..." 
Keep each bullet point under 15 words.""",

    "student": """You are a medical education assistant helping a pharmacy or medical student.
Structure your response with these exact sections:
**Overview** — drug class and mechanism of action
**Key Indications** — what conditions it treats
**Important Considerations** — dosing, contraindications, notable interactions
Use standard pharmacological terms but briefly explain any complex concepts.""",

    "clinician": """You are a clinical reference assistant for a licensed healthcare provider.
Be exhaustive and precise. Include:
- Drug class and full mechanism of action
- All contraindications with severity
- Drug-drug and drug-disease interactions
- Dosing details including renal/hepatic adjustments if relevant
- Key monitoring parameters
Use standard clinical notation. Do not simplify."""
}


# ==========================================================
# NEW: APPLY PERSONA FUNCTION
# Takes the raw drug dict from query_drug() and rewrites
# it using Groq based on the selected persona.
# ==========================================================

def apply_persona(drug_data: dict, question: str, persona: str = "patient") -> dict:

    # Get the system prompt for the selected persona.
    # Falls back to patient if an unknown persona is passed.
    system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["patient"])

    # Format the raw drug data as context for the LLM.
    # This is what was retrieved from your CSV / ChromaDB.
    context = f"""
Drug: {drug_data.get('drug', '')}
Uses: {drug_data.get('uses', '')}
Dosage: {drug_data.get('dosage', '')}
Side Effects: {drug_data.get('side_effects', '')}
Warnings: {drug_data.get('warnings', '')}
Interactions: {drug_data.get('interactions', '')}
Contraindications: {drug_data.get('contraindications', '')}
""".strip()

    # Build the user message combining context + original question
    user_message = f"""Drug label information:
{context}

Question: {question}

Answer using ONLY the information above. 
If something is not mentioned in the label, say "Not mentioned in the label."
"""

    llm = get_llm()

    # Pass system prompt + user message to Groq
    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    # Return both the structured raw data AND the persona-rewritten answer.
    # This lets your frontend show the LLM answer while keeping raw data available.
    return {
        "drug": drug_data.get("drug"),
        "persona": persona,
        "answer": response.content,       # persona-rewritten response shown to user
        "raw": drug_data                   # original structured data, useful for debugging
    }


# ==========================================================
# EXISTING: GROQ LLM SETUP (UNCHANGED)
# ==========================================================

def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


# ==========================================================
# EXISTING: CLINICAL QUERY (UNCHANGED)
# Still used by the /ask/prescription/clinical endpoint.
# ==========================================================

def clinical_query(question: str):

    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )

    if not results["documents"] or not results["documents"][0]:
        return {"message": "No relevant drug information found."}

    context = "\n\n".join(results["documents"][0])

    llm = get_llm()

    prompt = f"""
You are a clinical drug safety assistant.

Use ONLY the provided drug label context.
If answer is not present, say:
"Not mentioned in the provided label."

Context:
{context}

Question:
{question}

Respond strictly in this format:

Safety Assessment:
Reason:
Source Evidence:
"""

    response = llm.invoke(prompt)

    return {
        "question": question,
        "analysis": response.content
    }