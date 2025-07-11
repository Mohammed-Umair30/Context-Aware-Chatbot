import os
import shutil
import time
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import wikipedia

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_and_split(file_path, file_ext):
    if file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Upload PDF or TXT.")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vectorstore(documents):
    persist_directory = f"chroma_db_{int(time.time())}"
    vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb, persist_directory

def load_wikipedia_page(query):
    page_content = wikipedia.page(query).content
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([page_content])

def cleanup_old_vectorstores(base_dir="."):
    for folder in os.listdir(base_dir):
        if folder.startswith("chroma_db_") and os.path.isdir(folder):
            try:
                shutil.rmtree(os.path.join(base_dir, folder))
            except Exception as e:
                print(f"⚠️ Could not remove {folder}: {e}")
