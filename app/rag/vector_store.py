import faiss
import pickle

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, vectors, texts):
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, vector, k: int):
        _, idx = self.index.search(vector, k)
        return [self.texts[i] for i in idx[0]]

    def save(self, path):
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, path):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/texts.pkl", "rb") as f:
            self.texts = pickle.load(f)
