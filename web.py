import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from preprocess import normalize_text
from intent_recognizer import detect_intent
from rag_engine import load_chunks, build_index, semantic_search, ask_gpt_fallback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit page setup ---
st.set_page_config(page_title="🎓 CrescentBot", layout="centered")
st.title("🎓 CrescentBot - Your University Assistant")

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load model and build index ---
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    questions, data = load_chunks()
    index, _ = build_index(questions, model)
    return model, index, questions, data

model, index, questions, data = load_model_and_index()

# --- Chat input ---
user_input = st.chat_input("Ask CrescentBot anything about the university...")

if user_input:
    with st.spinner("Thinking..."):

        # Step 1: Preprocess
        norm_query = normalize_text(user_input)

        # Step 2: Intent detection
        intent = detect_intent(norm_query)

        # Step 3: Search with FAISS
        result, score = semantic_search(norm_query, model, index, questions, data)

        # Step 4: GPT fallback if needed
        if score > 1.2 or intent == "unknown":
            # Pass last 3 Q&A as memory to GPT
            context = "\n".join([f"Q: {x['user']}\nA: {x['bot']}" for x in st.session_state.chat_history[-3:]])
            response = ask_gpt_fallback(norm_query, context=context)
        else:
            response = result["answer"]

        # Step 5: Save to session memory
        st.session_state.chat_history.append({"user": user_input, "bot": response})

# --- Display conversation ---
# Display full chat history with a typing effect
for chat in st.session_state.chat_history[:-1]:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])

# Show the last message with typing animation
if st.session_state.chat_history:
    last = st.session_state.chat_history[-1]
    with st.chat_message("user"):
        st.markdown(last["user"])
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for char in last["bot"]:
            full_text += char
            placeholder.markdown(full_text + "▌")  # blinking cursor
            time.sleep(0.015)
        placeholder.markdown(full_text)  # final message without cursor

