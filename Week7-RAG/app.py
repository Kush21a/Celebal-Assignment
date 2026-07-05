import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

from utils import create_vector_db, embedding

st.set_page_config(
    page_title="Document Question Answering",
    page_icon="📄"
)

st.title("📄 Document Question Answering System (RAG)")
st.write("Upload one or more PDF documents and ask questions about them.")

# -------------------------
# Upload PDFs
# -------------------------

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    os.makedirs("docs", exist_ok=True)

    # Save uploaded PDFs
    for file in uploaded_files:

        with open(os.path.join("docs", file.name), "wb") as f:
            f.write(file.getbuffer())

    with st.spinner("Creating Vector Database..."):

        create_vector_db()

    st.success("✅ PDF(s) uploaded and indexed successfully!")

st.divider()

# -------------------------
# Ask Question
# -------------------------

question = st.text_input("Ask a question about your document")

if st.button("Ask"):

    if not os.path.exists("data/faiss_index"):

        st.error("Please upload a PDF first.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        db = FAISS.load_local(
            "data/faiss_index",
            embedding,
            allow_dangerous_deserialization=True
        )

        retriever = db.as_retriever(
            search_kwargs={"k": 3}
        )

        llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

        with st.spinner("Searching document..."):

            result = qa.invoke(
                {"query": question}
            )

        st.subheader("Answer")

        st.success(result["result"])