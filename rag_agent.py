import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY from .env:", api_key)

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


def initialize_rag():
    # Load PDF (ensure this path is correct and relative to where you run the app)
    loader = PyPDFLoader("data/Prospectus2025-26.pdf")
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Google Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Gemini LLM (Chat model)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    # Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    return qa_chain
