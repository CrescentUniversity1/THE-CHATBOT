import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from symspellpy import SymSpell
from dotenv import load_dotenv
from datetime import datetime
import re
import time

from log_utils import log_query  # <-- NEW for analytics logging

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="🎓 Crescent Uni Assistant",
    layout="wide"
)

st.markdown("<h2 style='text-align:center;'>🎓 CrescentBot - Your University Assistant</h2>", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_resource
def load_chunks():
    with open("data/qa_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

data = load_chunks()
questions = [item["question"] for item in data]

# --- Load Embeddings and FAISS Index ---
@st.cache_resource
def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(questions, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index

model, index = build_index()

# --- Load Abbreviation Map ---
@st.cache_data
def load_abbreviation_map():
    with open("data/abbreviation.json", "r", encoding="utf-8") as f:
        return json.load(f)

abbreviation_map = load_abbreviation_map()

# --- SymSpell for Spell Correction ---
@st.cache_resource
def load_symspell():
    symspell = SymSpell(max_dictionary_edit_distance=2)
    symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
    return symspell

symspell = load_symspell()

# --- Helpers ---
def expand_abbreviations(text):
    words = text.split()
    return ' '.join([abbreviation_map.get(w.lower(), w) for w in words])

def correct_spelling(text):
    suggestions = symspell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def normalize_text(text):
    text = expand_abbreviations(text)
    text = correct_spelling(text)
    return text.lower()

def detect_intent(text):
    greetings = ["hi", "hello", "good morning", "good afternoon"]
    farewells = ["bye", "goodbye", "see you"]
    if any(g in text.lower() for g in greetings):
        return "greeting"
    elif any(f in text.lower() for f in farewells):
        return "farewell"
    elif re.search(r"\b(thank(s)?|appreciate)\b", text.lower()):
        return "gratitude"
    return "unknown"

def semantic_search(query, model, index, questions, data, top_k=1):
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)
    if I[0][0] < len(questions):
        return data[I[0][0]], D[0][0]
    return None, float("inf")

def ask_gpt_fallback(prompt, context=None):
    try:
        system_msg = {
            "role": "system",
            "content": (
                "You are CrescentBot, a helpful and empathetic AI assistant for Crescent University. "
                "Your job is to provide clear, friendly, and accurate answers to student queries "
                "related to courses, departments, staff, and campus life. Be conversational and natural."
            )
        }
        messages = [system_msg]
        if context:
            messages.append({"role": "system", "content": f"Previous context:\n{context}"})
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.4,
            max_tokens=500,
            timeout=15
        )
        return response.choices[0].message["content"].strip()

    except Exception:
        return "Sorry, I'm currently unable to fetch a response from GPT-4."

def type_response(response):
    placeholder = st.empty()
    typed = ""
    for char in response:
        typed += char
        placeholder.markdown(f"<div style='font-size:18px;'>{typed}▌</div>", unsafe_allow_html=True)
        time.sleep(0.01)
    placeholder.markdown(f"<div style='font-size:18px;'>{typed}</div>", unsafe_allow_html=True)

# --- Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat UI ---
with st.container():
    for chat in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {chat['user']}")
        st.markdown(f"**🤖 CrescentBot:** {chat['bot']}")

user_input = st.text_input("Ask me anything about Crescent University...", key="input")

if user_input:
    with st.spinner("Thinking..."):
        norm_query = normalize_text(user_input)
        intent = detect_intent(norm_query)
        result, score = semantic_search(norm_query, model, index, questions, data)
        use_fallback = score > 1.2 or intent == "unknown"

        if use_fallback:
            context = "\n".join([f"Q: {x['user']}\nA: {x['bot']}" for x in st.session_state.chat_history[-3:]])
            response = ask_gpt_fallback(norm_query, context=context)
        else:
            response = result["answer"]

        st.session_state.chat_history.append({"user": user_input, "bot": response})
        type_response(response)

        # Log query
        log_query({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_input": user_input,
            "normalized_input": norm_query,
            "intent": intent,
            "matched_question": result["question"] if not use_fallback else None,
            "faiss_score": float(score),
            "used_gpt_fallback": use_fallback,
            "response": response,
            "feedback": None
        })
