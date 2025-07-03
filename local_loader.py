import os
from pathlib import Path

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


def list_txt_files(data_dir="./data"):
    paths = Path(data_dir).glob('**/*.txt')
    for path in paths:
        yield str(path)


def load_txt_files(data_dir="./data"):
    docs = []
    paths = list_txt_files(data_dir)
    for path in paths:
        print(f"Loading {path}")
        try:
            # Try UTF-8 encoding first
            loader = TextLoader(path, encoding='utf-8')
            docs.extend(loader.load())
        except UnicodeDecodeError:
            try:
                # Fall back to latin-1 if UTF-8 fails
                print(f"UTF-8 failed for {path}, trying latin-1")
                loader = TextLoader(path, encoding='latin-1')
                docs.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    return docs


def load_csv_files(data_dir="./data"):
    docs = []
    paths = Path(data_dir).glob('**/*.csv')
    for path in paths:
        loader = CSVLoader(file_path=str(path))
        docs.extend(loader.load())
    return docs


# Use with result of file_to_summarize = st.file_uploader("Choose a file") or a string.
# or a file like object.
def get_document_text(uploaded_file, title=None):
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)
    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page = page.extract_text()
            doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)

    else:
        # assume text - handle encoding issues
        try:
            doc_text = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                doc_text = uploaded_file.read().decode('latin-1')
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Reset file pointer
                doc_text = uploaded_file.read().decode('cp1252', errors='ignore')
        
        doc = Document(page_content=doc_text, metadata={'title': title})
        docs.append(doc)

    return docs


if __name__ == "__main__":
    example_pdf_path = "examples/healthy_meal_10_tips.pdf"
    docs = get_document_text(open(example_pdf_path, "rb"))
    for doc in docs:
        print(doc)
    docs = get_document_text(open("examples/us_army_recipes.txt", "rb"))
    for doc in docs:
        print(doc)
    txt_docs = load_txt_files("examples")
    for doc in txt_docs:
        print(doc)
    csv_docs = load_csv_files("examples")
    for doc in csv_docs:
        print(doc)

