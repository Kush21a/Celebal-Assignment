from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Folder containing PDFs
DOCS_PATH = "docs"

# Folder to save the vector database
DB_PATH = "data/faiss_index"

# Load all PDFs
documents = []

for file in os.listdir(DOCS_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages.")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store embeddings in FAISS
vectorstore = FAISS.from_documents(chunks, embedding_model)

vectorstore.save_local(DB_PATH)

print("FAISS index saved successfully!")