from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFaceHub # Consider HuggingFace model
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_ollama import OllamaLLM

from config import Config
## Test


def get_model():
    if Config.MAIN_MODEL == "Groq":
        chat_model = ChatGroq(temperature=Config.MODEL_TEMP,model=Config.MODEL_NAME, 
                              api_key=Config.MODEL_KEY)
    else:
        chat_model = OllamaLLM(model=Config.MODEL_NAME, base_url=Config.OLLAMA_HOST,
                               temperature=Config.MODEL_TEMP)
    return chat_model