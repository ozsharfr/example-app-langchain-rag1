from sentence_transformers import SentenceTransformer
from config import Config
from langchain_core.documents import Document 
from typing import Union , List

class SentenceTransformerEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_tensor=False).tolist()
def get_embeddings():
    if Config.EMBEDDING_MODEL == 'TRANSFORMER':
        return SentenceTransformerEmbeddings(Config.TRANSFORMER_MODEL)   
# def embed_text(text_or_doc: Union [str, Document]) -> list[float]:
#     model = SentenceTransformer(Config.TRANSFORMER_MODEL)
    
#     texts = get_texts(text_or_docs=text_or_doc)
#     embedding = model.encode(texts, convert_to_tensor=True)
#     return embedding , texts