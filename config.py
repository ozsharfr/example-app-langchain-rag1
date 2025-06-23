import os
import logging
from dotenv import load_dotenv

load_dotenv()

class Config:
    FILE_PATH = os.getenv("FILE_PATH", "abstracts_park.txt")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
    MAIN_MODEL = os.getenv("MAIN_MODEL", "Groq")
    MODEL_NAME_GCP = os.getenv("MODEL_NAME_GCP", "gpt-3.5-turbo")
    MODEL_TEMP = float(os.getenv("MODEL_TEMP", 0))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "TRANSFORMER")
    RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 5))
    DISTANCE_METHOD = os.getenv("DISTANCE_METHOD", 'cosine')
    MIN_RELEVANCE_SCORE = int(os.getenv("MIN_RELEVANCE_SCORE", 5))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 300))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    EMAIL_FOR_PUBMED = os.getenv("EMAIL_FOR_PUBMED", "")
    TRANSFORMER_MODEL = os.getenv("TRANSFORMER_MODEL", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "app.log"
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    DATABASE = os.getenv("DATABASE", "Chroma")
    MODEL_KEY = os.getenv("MODEL_KEY", "")

    @staticmethod
    def setup_logging():
        logging.basicConfig(
            filename=Config.LOG_FILE,
            level=Config.LOG_LEVEL,
            format=Config.LOG_FORMAT,
            filemode="a"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        logging.getLogger().addHandler(console_handler)

Config.setup_logging()


    