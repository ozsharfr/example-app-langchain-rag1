from config import Config

def get_vector_store_class():
    store_type = Config.DATABASE.lower()
    if store_type == 'chroma':
        from langchain_community.vectorstores import Chroma
        return Chroma
    elif store_type == 'faiss':
        from langchain_community.vectorstores import FAISS
        return FAISS
    # Add more stores as needed
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")