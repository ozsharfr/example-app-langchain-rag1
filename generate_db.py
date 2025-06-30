from remote_loader import get_wiki_docs
from local_loader import get_document_text
from splitter import split_documents
from time import sleep
from typing import List
import logging
from config import Config
import os

from define_embeddings import get_embeddings
from define_db import get_vector_store_class

EMBED_DELAY = 0.02  # 20 milliseconds

# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)

from langchain_core.documents import Document

# This happens all at once, not ideal for large datasets.
def create_vector_db(texts,db_name , embeddings=None):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    # Select embeddings
    if not embeddings:
        # Initialize Selected Embeddings
        embeddings = get_embeddings()

    proxy_embeddings = EmbeddingProxy(embeddings) ##?
    # Create a vectorstore from documents
    # this will be a chroma collection with a default name.
    selected_db = get_vector_store_class()
    db = selected_db(collection_name=Config.DATABASE.lower(),
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join("store/", db_name),
                collection_metadata={"hnsw:space": Config.DISTANCE_METHOD})
    
    db.add_documents(texts)
    db.persist()
    return db

def append_to_vector_db(texts,db_name , embeddings=None):
    db = load_vector_db(db_name ,embeddings)
    db.add_documents(texts)
    db.persist()
    return db

def load_vector_db(db_name , embeddings=None):
    if not embeddings:
        embeddings = get_embeddings()
    proxy_embeddings = EmbeddingProxy(embeddings)
    selected_db = get_vector_store_class()
    db = selected_db(
        collection_name=db_name,
        embedding_function=proxy_embeddings,
        persist_directory=os.path.join("store/", db_name ),
        collection_metadata={"hnsw:space": Config.DISTANCE_METHOD}
    )
    return db  # This will load the existing persisted DB

from remote_loader import main as get_docs
def main(db_name ,append = False ):
    # connector (could be multiple connectors) :
    #docs = get_docs()#
    docs = get_wiki_docs(query="Bertrand Russell", load_max_docs=Config.RETRIEVE_TOP_K) ## change
    # Split to chunks
    texts = split_documents(docs)
    # Either append or create new DB
    if append:
        db = append_to_vector_db(texts , db_name)
    else:
        db = create_vector_db(texts, db_name)


if __name__ =='__main__':
    main(db_name=Config.DATABASE , append = False)