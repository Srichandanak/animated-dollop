
# # rag.py

# import os
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.config import Settings
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# load_dotenv()
# # rag.py — add this after load_dotenv()

# import json
# from pathlib import Path
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # only once, at top
# # BASE_DIR = "D:/animated-dollop/drug_encyclopedia_api/project"
# # Load drug map for brand → generic resolution
# _drug_map_path = os.path.join(BASE_DIR, "drug_map.json")
# if os.path.exists(_drug_map_path):
#     DRUG_MAP = json.loads(Path(_drug_map_path).read_text())
# else:
#     # fallback empty structure if file doesn't exist yet
#     DRUG_MAP = {"brand_to_generic": {}, "generics": {}}
# # -------------------------
# # LOAD DATA (UNCHANGED)
# # -------------------------

# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# RX_PATH = os.path.join(BASE_DIR, "prescription_structured.csv")
# HOMEO_PATH = os.path.join(BASE_DIR, "homeopathic_structured.csv")

# rx_df = pd.read_csv(RX_PATH).fillna("")
# homeo_df = pd.read_csv(HOMEO_PATH).fillna("")

# # -------------------------
# # LOAD EMBEDDING MODEL (UNCHANGED)
# # -------------------------

# model = SentenceTransformer("all-MiniLM-L6-v2")

# # -------------------------
# # INIT CHROMA (UNCHANGED)
# # -------------------------

# chroma_path = os.path.join(BASE_DIR, "chroma_store")

# client = chromadb.PersistentClient(path=chroma_path)

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
#         metadatas.append({"drug": row["drug"], "type": "prescription"})
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
#         metadatas.append({"drug": row["drug"], "type": "homeopathic"})
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

# def query_drug(query: str, source: str, persona: str = "patient"):
#     # NOTE: persona is accepted here and passed to the
#     # persona layer AFTER the drug data is retrieved.
#     # The retrieval logic below is completely unchanged.

#     query_lower = query.lower()

#     if source == "prescription":
#         df = rx_df
#     elif source == "homeopathic":
#         df = homeo_df
#     else:
#         return {"message": "Invalid source"}

#     # --- exact match first (unchanged) ---
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

#             # NEW: once we have the raw drug data, run it through
#             # the persona layer before returning
#             return apply_persona(drug_data=response, question=query, persona=persona)

#     # --- semantic fallback (unchanged) ---
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

#     # NEW: same here — apply persona before returning semantic fallback result
#     return apply_persona(drug_data=response, question=query, persona=persona)


# # ==========================================================
# # NEW: PERSONA SYSTEM PROMPTS
# # One prompt per persona. These control how Groq rewrites
# # the raw drug data retrieved above.
# # ==========================================================

# PERSONA_PROMPTS = {
#     "patient": """You are a friendly health assistant explaining medication to a patient with no medical background.
# Use only plain English. No Latin, no abbreviations, no jargon.
# Format your response as:
# - One simple sentence saying what the drug is for
# - 3 to 5 bullet points covering key information
# - One line starting with "Ask your doctor if..." 
# Keep each bullet point under 15 words.""",

#     "student": """You are a medical education assistant helping a pharmacy or medical student.
# Structure your response with these exact sections:
# **Overview** — drug class and mechanism of action
# **Key Indications** — what conditions it treats
# **Important Considerations** — dosing, contraindications, notable interactions
# Use standard pharmacological terms but briefly explain any complex concepts.""",

#     "clinician": """You are a clinical reference assistant for a licensed healthcare provider.
# Be exhaustive and precise. Include:
# - Drug class and full mechanism of action
# - All contraindications with severity
# - Drug-drug and drug-disease interactions
# - Dosing details including renal/hepatic adjustments if relevant
# - Key monitoring parameters
# Use standard clinical notation. Do not simplify."""
# }

# # ==========================================================
# # EXISTING: GROQ LLM SETUP (UNCHANGED)
# # ==========================================================

# def get_llm():
#     return ChatGroq(
#         model_name="llama-3.1-8b-instant",
#         temperature=0,
#         max_tokens=1024,
#         groq_api_key=os.getenv("GROQ_API_KEY")
#     )

# # ==========================================================
# # NEW: APPLY PERSONA FUNCTION
# # Takes the raw drug dict from query_drug() and rewrites
# # it using Groq based on the selected persona.
# # ==========================================================

# def apply_persona(drug_data: dict, question: str, persona: str = "patient") -> dict:

#     # Get the system prompt for the selected persona.
#     # Falls back to patient if an unknown persona is passed.
#     system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["patient"])

#     # Format the raw drug data as context for the LLM.
#     # This is what was retrieved from your CSV / ChromaDB.
#     context = f"""
# Drug: {drug_data.get('drug', '')}
# Uses: {drug_data.get('uses', '')}
# Dosage: {drug_data.get('dosage', '')}
# Side Effects: {drug_data.get('side_effects', '')}
# Warnings: {drug_data.get('warnings', '')}
# Interactions: {drug_data.get('interactions', '')}
# Contraindications: {drug_data.get('contraindications', '')}
# """.strip()

#     # Build the user message combining context + original question
#     user_message = f"""Drug label information:
# {context}

# Question: {question}

# Answer using ONLY the information above. 
# If something is not mentioned in the label, say "Not mentioned in the label."
# """

#     llm = get_llm()

#     # Pass system prompt + user message to Groq
#     from langchain_core.messages import SystemMessage, HumanMessage
#     response = llm.invoke([
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=user_message)
#     ])

#     # Return both the structured raw data AND the persona-rewritten answer.
#     # This lets your frontend show the LLM answer while keeping raw data available.
#     return {
#         "drug": drug_data.get("drug"),
#         "persona": persona,
#         "answer": response.content,       # persona-rewritten response shown to user
#         "raw": drug_data                   # original structured data, useful for debugging
#     }





# # ==========================================================
# # EXISTING: CLINICAL QUERY (UNCHANGED)
# # Still used by the /ask/prescription/clinical endpoint.
# # ==========================================================

# def clinical_query(question: str):

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
import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from chunker import build_all_chunks
from retriever import build_bm25_index, hybrid_retrieve

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Drug map ──────────────────────────────────────────────────────────────────
_drug_map_path = os.path.join(BASE_DIR, "drug_map.json")
DRUG_MAP = json.loads(Path(_drug_map_path).read_text()) if os.path.exists(_drug_map_path) \
    else {"brand_to_generic": {}, "generics": {}}

# ── Load data ─────────────────────────────────────────────────────────────────
rx_df    = pd.read_csv(os.path.join(BASE_DIR, "prescription_structured.csv")).fillna("")
homeo_df = pd.read_csv(os.path.join(BASE_DIR, "homeopathic_structured.csv")).fillna("")

# ── Embedding model ───────────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Chroma (fixed: PersistentClient) ─────────────────────────────────────────
chroma_path = os.path.join(BASE_DIR, "chroma_store")
client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_or_create_collection("drug_chunks")  # new name — chunks, not rows

# ── Build chunks + BM25 index (done once at startup) ─────────────────────────
ALL_CHUNKS = build_all_chunks(rx_df, homeo_df)
BM25_INDEX = build_bm25_index(ALL_CHUNKS)


# ── Indexing (runs only if collection is empty) ───────────────────────────────
# def index_data():
#     if collection.count() > 0:
#         return

#     texts = [c["text"] for c in ALL_CHUNKS]
#     metadatas = [{"drug": c["drug"], "type": c["type"]} for c in ALL_CHUNKS]
#     ids = [f"chunk_{i}" for i in range(len(ALL_CHUNKS))]
#     embeddings = model.encode(texts).tolist()

#     collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
#     print(f"Indexed {len(ALL_CHUNKS)} chunks.")
def index_data():
    if collection.count() > 0:
        return

    texts = [c["text"] for c in ALL_CHUNKS]
    metadatas = [{"drug": c["drug"], "type": c["type"]} for c in ALL_CHUNKS]
    ids = [f"chunk_{i}" for i in range(len(ALL_CHUNKS))]
    embeddings = model.encode(texts).tolist()

    # ChromaDB max batch size is 5461 — add in chunks of 5000
    BATCH_SIZE = 5000
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end]
        )
        print(f"Indexed {end}/{total} chunks...")

    print(f"Done. Total chunks indexed: {total}")

# ── LLM (defined before it's used) ───────────────────────────────────────────
def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


# ── Persona prompts ───────────────────────────────────────────────────────────
# Loaded from prompts.yaml (Phase 2 — see below)
import yaml

_prompts_path = os.path.join(BASE_DIR, "prompts.yaml")
with open(_prompts_path) as f:
    PROMPTS_CONFIG = yaml.safe_load(f)

PERSONA_PROMPTS = PROMPTS_CONFIG["personas"]


# ── Main query function ───────────────────────────────────────────────────────
def query_drug(query: str, source: str, persona: str = "patient") -> dict:
    chunks = hybrid_retrieve(
        query=query,
        collection=collection,
        model=model,
        chunks=ALL_CHUNKS,
        bm25_index=BM25_INDEX,
        source_filter=source,
        top_k=10,
        rerank_top_n=3
    )

    if not chunks:
        return {"message": "No relevant drug information found."}

    return apply_persona(chunks=chunks, question=query, persona=persona)


# ── Persona application ───────────────────────────────────────────────────────
def apply_persona(chunks: list[dict], question: str, persona: str = "patient") -> dict:
    system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["patient"])

    # Build cited context — each chunk gets a [1], [2], [3] label
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] (Drug: {chunk.get('drug','')}, Source: {chunk.get('type','')})\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    user_message = f"""Drug information (cite as [1], [2], [3]):
{context}

Question: {question}

Answer using ONLY the information above. After each claim, cite the chunk number like [1].
If something is not mentioned, say "Not mentioned in the label."
"""

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    return {
        "drug": chunks[0].get("drug") if chunks else None,
        "persona": persona,
        "answer": response.content,
        "sources": [{"chunk": i+1, "drug": c.get("drug"), "type": c.get("type")} for i, c in enumerate(chunks)]
    }


# ── Clinical query (unchanged endpoint, updated retrieval) ────────────────────
def clinical_query(question: str) -> dict:
    chunks = hybrid_retrieve(
        query=question,
        collection=collection,
        model=model,
        chunks=ALL_CHUNKS,
        bm25_index=BM25_INDEX,
        source_filter="prescription",
        top_k=10,
        rerank_top_n=3
    )

    if not chunks:
        return {"message": "No relevant drug information found."}

    context = "\n\n".join(c["text"] for c in chunks)
    llm = get_llm()

    prompt = PROMPTS_CONFIG["clinical_query"].format(context=context, question=question)
    response = llm.invoke(prompt)

    return {"question": question, "analysis": response.content}