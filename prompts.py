from langchain.prompts import ChatPromptTemplate , MessagesPlaceholder

def get_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    return prompt



from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_enriched_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are a helpful assistant. "
        "Use the provided context to answer the user's question as accurately as possible. "
        "If the answer is not in the context, say you don't know. "
        "Cite relevant sections or sources from the context when possible. "
        "Be concise and clear in your response."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", 
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "If you use information from the context, reference the section or source if available."
        )
    ])

    return prompt
