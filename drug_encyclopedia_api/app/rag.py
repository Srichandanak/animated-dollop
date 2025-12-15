from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

DATA_PATH = Path("data/spl_alltext_with_better_names.csv")

# Load data
df = pd.read_csv(DATA_PATH)
texts = df["full_text"].fillna("").tolist()
titles = df["title"].fillna("").tolist()

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # [web:153]

# Build embeddings once at startup
embs = model.encode(texts, convert_to_numpy=True)

# Chroma client & collection
client = chromadb.Client()
collection = client.create_collection("dailymed_homeopathic_ui")

collection.add(
    ids=[str(i) for i in range(len(df))],
    documents=texts,
    metadatas=[{"title": t} for t in titles],
    embeddings=embs.tolist(),
)

def search_labels(query: str, top_k: int = 5):
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    res = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
    )
    results = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        results.append(
            {
                "title": meta.get("title", ""),
                "text": doc,
            }
        )
    return results
