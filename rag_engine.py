import faiss
import json
import numpy as np
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
