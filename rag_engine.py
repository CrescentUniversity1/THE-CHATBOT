import faiss
import json
import numpy as np
import openai
import os
from sentence_transformers import SentenceTransformer

# Load dataset
def load_chunks(path="data/qa_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    return questions, data

# Build FAISS index
def build_index(questions, model):
    embeddings = model.encode(questions, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Perform search
def semantic_search(query, model, index, questions, data, top_k=1):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "question": questions[idx],
            "answer": data[idx]["answer"]
        })

    return results[0], distances[0][0]

openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt_fallback(query, context=None):
    base_prompt = (
        "You are CrescentBot, the university assistant for Crescent University. "
        "Answer the student's question clearly, based only on the school's official info. "
        "Be concise, and if the question is outside your scope, say you don’t have the answer.\n\n"
    )

    full_prompt = base_prompt
    if context:
        full_prompt += f"Context:\n{context}\n\n"

    full_prompt += f"Student's Question: {query}\n\nCrescentBot:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can switch to "gpt-3.5-turbo"
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I'm currently unable to fetch a response from GPT-4."
