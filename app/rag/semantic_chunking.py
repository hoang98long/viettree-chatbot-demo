import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semantic_chunk_by_heading(text: str):
    """
    Chia theo heading in hoa / dòng bắt đầu bằng số / bullet
    """
    chunks = []
    current_chunk = []

    lines = text.splitlines()

    for line in lines:
        if re.match(r"^[A-ZÀ-Ỹ0-9\s]{5,}$", line.strip()):
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = []
        current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    return [c for c in chunks if len(c) > 50]

def semantic_chunk_by_similarity(
    paragraphs,
    embeddings,
    threshold=0.75
):
    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        sim = cosine_similarity(
            embeddings[i-1:i],
            embeddings[i:i+1]
        )[0][0]

        if sim >= threshold:
            current_chunk.append(paragraphs[i])
        else:
            chunks.append("\n".join(current_chunk))
            current_chunk = [paragraphs[i]]

    chunks.append("\n".join(current_chunk))
    return chunks
