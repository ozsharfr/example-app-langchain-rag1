from langchain.prompts import ChatPromptTemplate

def get_judging_prompt():
    return ChatPromptTemplate.from_messages([
        ("human", "Evaluate the following answer based on relevance, accuracy, and completeness:\n\n"
                  "Question: {question}\n\n"
                  "Answer: {answer}\n\n"
                  "Sources: {sources}\n\n"
                  "Provide a score (1-10) and a brief explanation.")
    ])

def judge_answer(llm, question, answer, sources):
    judging_prompt = get_judging_prompt()
    formatted_prompt = judging_prompt.format(
        question=question,
        answer=answer,
        sources=sources
    )
    response = llm.invoke(formatted_prompt)
    return response

