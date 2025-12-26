from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class EmbeddingModel:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()

    def encode(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()

def load_embedding_model(model_path: str):
    return EmbeddingModel(model_path)
