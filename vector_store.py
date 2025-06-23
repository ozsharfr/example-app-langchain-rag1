import logging
import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from local_loader import get_document_text
from remote_loader import download_file
from splitter import split_documents
from dotenv import load_dotenv
from time import sleep

from define_embeddings import get_embeddings
from define_db import get_vector_store_class

EMBED_DELAY = 0.02  # 20 milliseconds

from config import Config
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



# This happens all at once, not ideal for large datasets.
def create_vector_db(texts, embeddings=None, collection_name="chroma"):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    # Select embeddings
    if not embeddings:
        # Initialize Selected Embeddings
        embeddings = get_embeddings()

    proxy_embeddings = EmbeddingProxy(embeddings)
    # Create a vectorstore from documents
    # this will be a chroma collection with a default name.
    selected_db = get_vector_store_class()
    db = selected_db(collection_name=Config.DATABASE.lower(),
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join("store/", Config.DATABASE.lower()),
                collection_metadata={"hnsw:space": Config.DISTANCE_METHOD})
    
    db.add_documents(texts)

    return db


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def main():
    load_dotenv()

    pdf_filename = "examples/mal_boole.pdf"

    if not os.path.exists(pdf_filename):
        math_analysis_of_logic_by_boole = "https://www.gutenberg.org/files/36884/36884-pdf.pdf"
        local_pdf_path = download_file(math_analysis_of_logic_by_boole, pdf_filename)
    else:
        local_pdf_path = pdf_filename

    print(f"PDF path is {local_pdf_path}")

    with open(local_pdf_path, "rb") as pdf_file:
        docs = get_document_text(pdf_file, title="Analysis of Logic")

    texts = split_documents(docs)
    vs = create_vector_db(texts)

    results = find_similar(vs, query="What is meant by the simple conversion of a proposition?")
    MAX_CHARS = 300
    print("=== Results ===")
    for i, text in enumerate(results):
        # cap to max length but split by words.
        content = text.page_content
        n = max(content.find(' ', MAX_CHARS), MAX_CHARS)
        content = text.page_content[:n]
        print(f"Result {i + 1}:\n {content}\n")


if __name__ == "__main__":
    main()
