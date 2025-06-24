from langchain.memory import ConversationBufferMemory
import os
from typing import List, Iterable, Any

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from memory import SimpleTextRetriever 
from langchain_core.runnables.history import RunnableWithMessageHistory
from basic_chain import get_model
from rag_chain import make_rag_chain

os.environ["LANGCHAIN_TRACING_V2"] = "false"

def create_chat_with_memory(llm, rag_chain, memory):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | rag_chain
    with_memory = RunnableWithMessageHistory(
        chain,
        lambda session_id: memory,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return with_memory


def main():
    load_dotenv()
    model = get_model()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    system_prompt = "You are a helpful AI assistant for busy professionals trying to improve their health."
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    text_path = "examples/grocery.md"
    text = open(text_path, "r").read()
    retriever = SimpleTextRetriever.from_texts([text])
    rag_chain = make_rag_chain(model, retriever, rag_prompt=None)
    chain = create_chat_with_memory(model, rag_chain, memory.chat_memory) | StrOutputParser()

    queries = [
        "What do I need to get from the grocery store besides milk?",
        "Which of these items can I find at a farmer's market?",
    ]

    for query in queries:
        print(f"\nQuestion: {query}")
        response = chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": "foo"}}
        )
        print(f"Answer: {response}")

if __name__ == '__main__':
    main()
