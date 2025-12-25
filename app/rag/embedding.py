from sentence_transformers import SentenceTransformer

def load_embedding_model(model_path: str):
    return SentenceTransformer(model_path)
