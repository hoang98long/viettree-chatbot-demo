from app.rag.pipeline import retrieve_chunks, generate_answer
from app.rag.prompts import REACT_PROMPT


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    return REACT_PROMPT.format(
        context=context,
        question=question
    )


def extract_refined_query(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("NEED_MORE_INFO"):
            return line.replace("NEED_MORE_INFO:", "").strip()
    return None


def clean_final_answer(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not line.startswith("NEED_MORE_INFO"):
            lines.append(line)
    return "\n".join(lines).strip()


def agentic_answer(question: str) -> str:
    """
    Agentic RAG nhẹ:
    - Retrieve tối đa 2 lần
    """

    # ===== Retrieve lần 1 =====
    chunks = retrieve_chunks(question)
    prompt = build_prompt(question, chunks)
    result = generate_answer(prompt)

    refined_query = extract_refined_query(result)

    # ===== Adaptive retrieve lần 2 (nếu cần) =====
    if refined_query:
        chunks = retrieve_chunks(refined_query)
        prompt = build_prompt(question, chunks)
        result = generate_answer(prompt)

    return clean_final_answer(result)
