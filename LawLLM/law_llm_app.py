import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Load API key ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  # change if using OpenAI
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- Streamlit Setup ---
st.set_page_config(page_title="Better Call Advoca!", page_icon="⚖️", layout="wide")

# --- Custom CSS for ChatGPT-like style ---
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 900px;
        margin: auto;
    }
    .stChatMessage {
        padding: 0.8rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #e9f5ff;
        border: 1px solid #b6e0ff;
    }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #f7f7f8;
        border: 1px solid #e5e5e5;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("Better Call Advoca!")
st.caption("Your AI-powered legal assistant (⚠️ This is not professional legal advice).")

# --- Load FAISS index and documents ---
if "faiss_index" not in st.session_state:
    with open("law_cases.txt", "r", encoding="utf-8") as f:
        st.session_state.documents = [line.strip() for line in f if line.strip()]

    st.session_state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = st.session_state.embedder.encode(st.session_state.documents).astype("float32")

    st.session_state.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    st.session_state.faiss_index.add(embeddings)

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
if user_input := st.chat_input("Ask Advoca anything about law..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- FAISS Retrieval ---
    query_vec = st.session_state.embedder.encode([user_input]).astype("float32")
    D, I = st.session_state.faiss_index.search(query_vec, k=3)
    retrieved_cases = [st.session_state.documents[i] for i in I[0]]
    retrieval_text = "\n\n".join(retrieved_cases)

    # --- LLM Call (Gemini) ---
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful legal assistant called Better Call Advoca!. "
                        "Use the following retrieved legal cases to answer the question. "
                        "Always include a disclaimer: 'This is not legal advice.'\n\n"
                        f"Retrieved cases:\n{retrieval_text}"
                    ),
                },
                *st.session_state["messages"],
            ],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
