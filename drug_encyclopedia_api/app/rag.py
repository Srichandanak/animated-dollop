# from pathlib import Path
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import chromadb
# import numpy as np

# DATA_PATH = Path("data/spl_alltext_with_better_names.csv")

# # Load data
# df = pd.read_csv(DATA_PATH)
# texts = df["full_text"].fillna("").tolist()
# titles = df["title"].fillna("").tolist()

# # Embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")  # [web:153]

# # Build embeddings once at startup
# embs = model.encode(texts, convert_to_numpy=True)

# # Chroma client & collection
# client = chromadb.Client()
# collection = client.create_collection("dailymed_homeopathic_ui")

# collection.add(
#     ids=[str(i) for i in range(len(df))],
#     documents=texts,
#     metadatas=[{"title": t} for t in titles],
#     embeddings=embs.tolist(),
# )

# def search_labels(query: str, top_k: int = 5):
#     q_emb = model.encode([query], convert_to_numpy=True)[0]
#     res = collection.query(
#         query_embeddings=[q_emb.tolist()],
#         n_results=top_k,
#     )
#     results = []
#     for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
#         results.append(
#             {
#                 "title": meta.get("title", ""),
#                 "text": doc,
#             }
#         )
#     return results
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import re

DATA_PATH = Path("data/spl_alltext_with_better_names.csv")

# Load data
df = pd.read_csv(DATA_PATH)

model = SentenceTransformer("all-MiniLM-L6-v2")

# client = chromadb.Client()
# collection = client.create_collection("dailymed_homeopathic_ui")
import chromadb
from chromadb.config import Settings

client = chromadb.Client(
    Settings(
        persist_directory="/opt/render/project/src/chroma_db",
        is_persistent=True
    )
)

collection = client.get_or_create_collection("spl_docs")


def split_sections(text: str):
    """
    Splits SPL text into sections using uppercase headers.
    Returns list of (section_name, section_text).
    """
    pattern = r"\n([A-Z][A-Z\s\-]+)\n"
    parts = re.split(pattern, text)

    sections = []

    # parts structure:
    # [intro, SECTION1, content1, SECTION2, content2, ...]
    if len(parts) < 3:
        return [("FULL_TEXT", text)]

    for i in range(1, len(parts), 2):
        section_name = parts[i].strip()
        section_text = parts[i + 1].strip()
        if section_text:
            sections.append((section_name, section_text))

    return sections


def build_collection_if_empty():
    existing = collection.count()

    if existing > 0:
        print(f"Collection already contains {existing} documents. Skipping build.")
        return

    print("Building collection from CSV...")

    documents = []
    metadatas = []
    ids = []

    counter = 0

    for _, row in df.iterrows():
        title = row["title"]
        file_name = row["file_name"]
        full_text = row["full_text"]

        sections = split_sections(full_text)

        for section_name, section_text in sections:
            documents.append(section_text)
            metadatas.append({
                "title": title,
                "file_name": file_name,
                "section": section_name
            })
            ids.append(str(counter))
            counter += 1

    embeddings = model.encode(documents, convert_to_numpy=True)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    print("Collection build complete.")


# Build once at startup
build_collection_if_empty()


def search_labels(query: str, top_k: int = 5, section=None, file_name=None):
    q_emb = model.encode([query], convert_to_numpy=True)[0]

    where = {}

    if section:
        where["section"] = section

    if file_name:
        where["file_name"] = file_name

    res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        where=where if where else None,
    )

    results = []

    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        results.append({
            "title": meta.get("title", ""),
            "file_name": meta.get("file_name", ""),
            "section": meta.get("section", ""),
            "text": doc,
        })

    return results


def get_available_filters():
    all_meta = collection.get(include=["metadatas"])["metadatas"]

    sections = sorted(set(m["section"] for m in all_meta))
    file_names = sorted(set(m["file_name"] for m in all_meta))

    return {
        "sections": sections,
        "file_names": file_names
    }
