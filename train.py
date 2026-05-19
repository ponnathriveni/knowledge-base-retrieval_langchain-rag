# Generated from: train.ipynb
# Converted at: 2026-05-19T06:12:26.394Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell
"""
!pip langchain-chains

!pip install langchain
!pip install langchain-community
!pip install langchain-core
!pip install langchain-text-splitters
!pip install langchain-huggingface
!pip install langchain-groq
!pip install faiss-cpu
!pip install pypdf
!pip install sentence-transformers"""



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

import os
os.environ["GROQ_API_KEY"] = "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"

os.environ["HF_TOKEN"] = "your_huggingface_token"

loader = PyPDFLoader("data.pdf")
documents = loader.load()
print(documents)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
docs

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
embeddings

db = FAISS.from_documents(docs, embeddings)
db

db.save_local("faiss_index")

print("Vector Database Saved Successfully")