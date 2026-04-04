# chunker.py
import pandas as pd
import os

def chunk_row(row: dict, source: str, chunk_size: int = 600, overlap: int = 100) -> list[dict]:
    """
    Converts one CSV row into overlapping text chunks of ~chunk_size chars.
    Each chunk keeps metadata so retrieval knows which drug/source it came from.
    """
    if source == "prescription":
        full_text = (
            f"Drug: {row.get('drug', '')}\n"
            f"Uses: {row.get('uses', '')}\n"
            f"Dosage: {row.get('dosage', '')}\n"
            f"Side Effects: {row.get('side_effects', '')}\n"
            f"Warnings: {row.get('warnings', '')}\n"
            f"Interactions: {row.get('interactions', '')}\n"
            f"Contraindications: {row.get('contraindications', '')}"
        )
    else:
        full_text = (
            f"Drug: {row.get('drug', '')}\n"
            f"Uses: {row.get('uses', '')}\n"
            f"Dosage: {row.get('dosage', '')}\n"
            f"Warnings: {row.get('warnings', '')}"
        )

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk_text = full_text[start:end]
        chunks.append({
            "text": chunk_text,
            "drug": row.get("drug", ""),
            "type": source,
        })
        start += chunk_size - overlap  # slide window with overlap

    return chunks


def build_all_chunks(rx_df: pd.DataFrame, homeo_df: pd.DataFrame) -> list[dict]:
    all_chunks = []
    for _, row in rx_df.iterrows():
        all_chunks.extend(chunk_row(row.to_dict(), "prescription"))
    for _, row in homeo_df.iterrows():
        all_chunks.extend(chunk_row(row.to_dict(), "homeopathic"))
    return all_chunks