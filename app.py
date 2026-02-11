import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from  langchain_classic.chains import  create_retrieval_chain
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

# Set API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("📄 RAG PDF Chatbot (Llama 3.3 70B)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load Llama 3.3 70B model from Groq
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    query = st.text_input("Ask a question about the PDF:")

    if query:
        response = qa_chain.run(query)
        st.write("### Answer:")
        st.write(response)
