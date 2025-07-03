import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory 
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.messages.base import BaseMessage
from define_model import get_model
from generate_db import load_vector_db
from prompts import get_prompt

from config import Config
import os

#from rag_chain import make_rag_chain
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain.callbacks.base import BaseCallbackHandler

class DocumentCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        self.retrieved_docs = []
    
    def on_retriever_end(self, documents, **kwargs):
        print(f"[Callback] Retrieved {len(documents)} docs")
        self.retrieved_docs = documents

def make_rag_chain(llm, retriever, prompt , memory):
    

    def retrieve_docs(input_data):
        # Use .invoke() to trigger callbacks
        docs = retriever.invoke(input_data["question"])  
        print(f"Docs found = {len(docs)}")
        input_data["_retrieved_docs"] = docs
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=retrieve_docs,
            question=lambda x: x["question"]
        )
        | prompt
        | llm
        | StrOutputParser()  
    )

    return RunnableWithMessageHistory(
        chain,
        lambda _: memory,
        input_messages_key="question",
        history_messages_key="chat_history"
    )


def main_memory(q : str , vs, memory ) -> tuple:
    load_dotenv()
    model = get_model()
    
    # Define callback handler
    doc_callback = DocumentCaptureCallback()
    prompt = get_prompt()
    # Besides similarly search, you can also use maximal marginal relevance (MMR) for selecting results.
    # retriever = vs.as_retriever(search_type="mmr")
    retriever = vs.as_retriever(callbacks=[doc_callback])
    # change : Add get_prompt
    #rag_chain = make_rag_chain(model, retriever, doc_callback=doc_callback, rag_prompt=None)
    rag_memory_chain = make_rag_chain(model, retriever, prompt, memory.chat_memory) 

    # Invoke with a dict
    response = rag_memory_chain.invoke(
        {"question": q},
        config={"configurable": {"session_id": "foo"},
            "callbacks": [doc_callback]
            }
    )
    
    # Extract document metadata (need to build docs with metadata)
    retrieved_docs = doc_callback.retrieved_docs
    
    return response , retrieved_docs

if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    vs = load_vector_db(db_name=Config.DATABASE.lower())
    questions = [
        "What were the most important contributions of Bertrand Russell to philosophy?",
        "What was the first book Bertrand Russell published?",
        "What was most notable about \"An Essay on the Foundations of Geometry\"?",
    ]
    for q in questions:
        answer , docs = main_memory(q=q, memory=memory , vs=vs)
        sources = '\n'.join([doc.metadata['source'] for doc in docs if 'source' in doc.metadata])
        print (answer)
        print (sources)
