from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_ensemble_retriever(vs, bm25_retriever, weights=[0.5, 0.5]):
    """Create an ensemble retriever combining BM25 and vector-based retrievers."""
    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs.as_retriever()],
        weights=weights
    )

    return ensemble_retriever

def initialize_bm25_retriever(texts):
    """Initialize and return a BM25 retriever from texts."""
    return BM25Retriever.from_texts([t.page_content for t in texts])
