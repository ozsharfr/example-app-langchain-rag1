from langchain.prompts import ChatPromptTemplate , MessagesPlaceholder

def get_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    return prompt