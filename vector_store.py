import logging
import os

from langchain_openai import OpenAIEmbeddings
from local_loader import get_document_text
from remote_loader import download_file
from splitter import split_documents
from dotenv import load_dotenv
from generate_db import create_vector_db

def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def main():
    load_dotenv()

    pdf_filename = "examples/mal_boole.pdf"

    if not os.path.exists(pdf_filename):
        math_analysis_of_logic_by_boole = "https://www.gutenberg.org/files/36884/36884-pdf.pdf"
        local_pdf_path = download_file(math_analysis_of_logic_by_boole, pdf_filename)
    else:
        local_pdf_path = pdf_filename

    print(f"PDF path is {local_pdf_path}")

    with open(local_pdf_path, "rb") as pdf_file:
        docs = get_document_text(pdf_file, title="Analysis of Logic")

    texts = split_documents(docs)
    vs = create_vector_db(texts)

    results = find_similar(vs, query="What is meant by the simple conversion of a proposition?")
    MAX_CHARS = 300
    print("=== Results ===")
    for i, text in enumerate(results):
        # cap to max length but split by words.
        content = text.page_content
        n = max(content.find(' ', MAX_CHARS), MAX_CHARS)
        content = text.page_content[:n]
        print(f"Result {i + 1}:\n {content}\n")


if __name__ == "__main__":
    main()
