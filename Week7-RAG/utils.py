import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vector_db():
    documents = []

    if not os.path.exists("docs"):
        os.makedirs("docs")

    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("docs", file))
            documents.extend(loader.load())

    if len(documents) == 0:
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embedding)

    os.makedirs("data", exist_ok=True)
    db.save_local("data/faiss_index")

    return True