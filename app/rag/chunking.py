def simple_chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    """
    Chunk text theo số ký tự (demo-friendly).
    chunk_size: độ dài chunk
    chunk_overlap: phần overlap giữa các chunk
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0

    return chunks
