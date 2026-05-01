from sentence_transformers import SentenceTransformer, util
import pickle
import torch
import requests
import json
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")

QUESTIONS_PATH = os.path.join(DATA_PATH, "questions.pkl")
ANSWERS_PATH = os.path.join(DATA_PATH, "answers.pkl")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
with open(QUESTIONS_PATH, "rb") as f:
    questions = pickle.load(f)

with open(ANSWERS_PATH, "rb") as f:
    answers = pickle.load(f)

# Precompute embeddings
question_embeddings = embed_model.encode(questions, convert_to_tensor=True)


def save_new_data(question, answer):
    """
    Save new question-answer pairs for incremental learning.
    """
    try:
        with open(DATA_JSON_PATH, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append({
        "question": question,
        "answer": answer
    })

    with open(DATA_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)


def generate_answer(prompt):
    """
    Generate answer using semantic search + TinyLlama.
    """
    try:
        prompt = prompt.lower().strip()

        # Step 1: Semantic similarity
        query_embedding = embed_model.encode(prompt, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, question_embeddings)

        index = torch.argmax(scores)
        score = scores[0][index].item()
        context = answers[index]

        # Step 2: Build prompt
        if score >= 0.25:
            final_prompt = f"""
You are a simple AI tutor.

Answer clearly in 3-4 lines.
Use simple words.
Give one short example.

Question: {prompt}
Reference: {context}

Answer:
"""
        else:
            final_prompt = f"""
You are a simple AI tutor.

Answer clearly in 3-4 lines.
Use simple words.
Give one short example.

Question: {prompt}

Answer:
"""

        # Step 3: Call TinyLlama via Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": final_prompt,
                "stream": False,
                "options": {
                    "num_predict": 120,
                    "temperature": 0.5
                }
            }
        )

        result = response.json().get("response", "").strip()

        # Step 4: Clean output
        if "Answer:" in result:
            result = result.split("Answer:")[-1].strip()

        # Step 5: Save learning (low confidence only)
        if score < 0.25:
            save_new_data(prompt, result)

        if not result:
            return "Sorry, I could not generate a proper answer."

        return result

    except Exception as e:
        return f"Error: {str(e)}"
