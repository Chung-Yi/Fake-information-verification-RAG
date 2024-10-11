from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.encoder = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.encoder.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        return self.encoder.encode(text).tolist()

# Initialize your wrapped encoder
encoder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")