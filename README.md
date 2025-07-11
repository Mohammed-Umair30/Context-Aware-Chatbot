# Context-Aware-Chatbot
# 🤖 Context-Aware Chatbot with Ollama + RAG

This project is a **context-aware conversational chatbot** built using [LangChain](https://python.langchain.com/), Ollama, and Retrieval-Augmented Generation (RAG).  
It allows users to ask questions based on their uploaded documents or Wikipedia pages, while maintaining conversation history and context.  

---

## 🚀 Features

- ✅ Upload **PDF** or **TXT** documents to build your own knowledge base.
- ✅ Or, load content directly from **Wikipedia** using a topic keyword.
- ✅ Chatbot can **remember conversation context** across multiple turns.
- ✅ Uses **vector embeddings** (via `HuggingFaceEmbeddings`) and **Chroma vector store** for fast, accurate retrieval.
- ✅ Integration with **Ollama LLM** (e.g., TinyLlama) for natural language answers.
- ✅ Professional system prompt ensures clear, precise, and well-structured responses.
- ✅ "Start New Chat" button to reset the conversation context anytime.
- ✅ "Reset vectorstore" button to clear previous data and upload new content.

---

## 🧰 Tech Stack

- **LangChain** (LangChain community modules and Ollama integration)
- **Hugging Face embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)
- **ChromaDB** for vector storage
- **Streamlit** for UI
- **Ollama** for local large language model inference

---

## 💻 How to Run

1️⃣ **Clone this repository**

```bash
git clone (https://github.com/Mohammed-Umair30/Context-Aware-Chatbot.git)
cd context-aware-chatbot-ollama-rag

