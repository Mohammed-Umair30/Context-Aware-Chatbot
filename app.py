import streamlit as st
import tempfile
import os
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from utils import load_and_split, create_vectorstore, load_wikipedia_page, embeddings, cleanup_old_vectorstores
from langchain_community.vectorstores import Chroma

# ---- Setup LLM ----
llm = ChatOllama(
    model="tinyllama:latest",
    model_kwargs={"num_predict": 300}
)

# ---- Streamlit UI ----
st.set_page_config(page_title="💬 Context-Aware Chatbot", page_icon="🤖")
st.title("🤖 Context-Aware Chatbot with Ollama + RAG")

# Sidebar
with st.sidebar:
    st.header("📄 Document Upload or Wikipedia")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    wiki_query = st.text_input("Or enter Wikipedia topic", "")

    if st.button("🗑️ Reset vectorstore (cleanup)"):
        cleanup_old_vectorstores()
        st.success("Old vectorstores cleaned up! Restart and upload new docs.")

    if st.button("🆕 Start New Chat"):
        st.session_state["messages"] = []
        if "qa_chain" in st.session_state:
            del st.session_state["qa_chain"]
        st.rerun()

# ---- Initialize ----
docs = []
persist_directory = None

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        docs.extend(load_and_split(tmp_file_path, file_ext))
        os.unlink(tmp_file_path)
    st.sidebar.success(f"Loaded {len(uploaded_files)} file(s).")

elif wiki_query:
    try:
        docs = load_wikipedia_page(wiki_query)
        st.sidebar.success(f"Wikipedia page loaded: {wiki_query}")
    except Exception as e:
        st.sidebar.error(f"Error loading Wikipedia page: {e}")

if docs:
    vectordb, persist_directory = create_vectorstore(docs)
else:
    existing_dirs = [d for d in os.listdir() if d.startswith("chroma_db_")]
    if existing_dirs:
        latest_dir = sorted(existing_dirs)[-1]
        vectordb = Chroma(persist_directory=latest_dir, embedding_function=embeddings)
        persist_directory = latest_dir
    else:
        vectordb = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Create chain only if not already in session
if "qa_chain" not in st.session_state and vectordb:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
    )

qa_chain = st.session_state.get("qa_chain", None)

# ---- Chat interface ----
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

prompt = st.chat_input("Ask your question...")

if prompt and qa_chain:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    system_instruction = (
        "You are a precise and helpful assistant. "
        "Answer concisely and accurately using the retrieved context. "
        "If unsure, say 'I am not sure' rather than guessing."
    )

    response = qa_chain({"question": f"{system_instruction}\n\n{prompt}"})
    answer = response["answer"]

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)
elif prompt and not qa_chain:
    st.warning("⚠️ Please upload documents or provide a Wikipedia topic before chatting.")
