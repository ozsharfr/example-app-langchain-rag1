# Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from config import Config

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False)

    contents = docs
    if docs and isinstance(docs[0], Document):
        contents = [doc.page_content for doc in docs]
    texts = text_splitter.create_documents(contents)

    # Enforce using documents
    texts = text_splitter.split_documents(docs)
    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts

if __name__ == '__main__':
    from generate_db import get_wiki_docs
    docs = get_wiki_docs(query="Bertrand Russell", load_max_docs=Config.RETRIEVE_TOP_K)
    split_documents(docs)