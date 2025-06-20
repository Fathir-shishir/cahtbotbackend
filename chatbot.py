import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, session
from flask_cors import CORS

# Configuration
GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "greetings"]
DEFAULT_RESPONSE = "I'm sorry, I don't have information about that. Please ask another question."
MIN_CONFIDENCE = 0.3  # Minimum similarity score to consider a match

# Load FAQ data
with open("faq_data.json", "r", encoding="utf-8") as file:
    faq_data = json.load(file)

# Initialize model and embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
questions = [item["question"] for item in faq_data]
question_embeddings = model.encode(questions)

# Normalize embeddings
norms = np.linalg.norm(question_embeddings, axis=1, keepdims=True)
question_embeddings_normalized = question_embeddings / norms

# FAISS index
dimension = question_embeddings_normalized.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(np.array(question_embeddings_normalized, dtype=np.float32))

app = Flask(__name__)
app.secret_key = "supersecretkey"
CORS(app)

def find_best_answers(query, top_k=5, margin=0.1):
    """Find matching questions with confidence threshold"""
    query_embedding = model.encode([query])
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    D, I = index.search(query_embedding_normalized.astype(np.float32), k=top_k)
    
    scores = D[0].tolist()
    indices = I[0].tolist()
    
    valid = [(idx, score) for idx, score in zip(indices, scores) if idx != -1 and score >= MIN_CONFIDENCE]
    if not valid:
        return []
    
    valid_indices, valid_scores = zip(*valid)
    top_score = valid_scores[0]
    
    candidates = []
    for idx, score in zip(valid_indices, valid_scores):
        if score >= (top_score - margin):
            candidates.append((int(idx), float(score)))
    
    return candidates

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query", "").strip().lower()

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Check for greetings
    if any(greet in user_query for greet in GREETINGS):
        return jsonify({"answer": "Hello! How can I help you today?"})

    # Session management
    if "conversation_history" not in session:
        session["conversation_history"] = []
    conversation_history = session["conversation_history"]

    # Handle pending options
    if "pending_options" in session:
        try:
            selected_num = int(user_query)
            pending_options = session["pending_options"]
            if 1 <= selected_num <= len(pending_options):
                selected_idx = pending_options[selected_num - 1]["index"]
                answer = faq_data[selected_idx]["answer"]
                conversation_history.append({"question": user_query, "answer": answer})
                session.pop("pending_options")
                return jsonify({"answer": answer})
            session.pop("pending_options")
            return jsonify({"answer": "Invalid selection. Please try again."})
        except ValueError:
            session.pop("pending_options")

    # Context handling
    if conversation_history and ("that" in user_query or "more" in user_query):
        last_topic = conversation_history[-1]["question"]
        user_query = f"{last_topic} {user_query}"

    # Process query
    candidates = find_best_answers(user_query)
    
    if not candidates:
        conversation_history.append({"question": user_query, "answer": DEFAULT_RESPONSE})
        session["conversation_history"] = conversation_history
        return jsonify({"answer": DEFAULT_RESPONSE})

    if len(candidates) == 1:
        best_idx = candidates[0][0]
        answer = faq_data[best_idx]["answer"]
        conversation_history.append({"question": user_query, "answer": answer})
        session["conversation_history"] = conversation_history
        return jsonify({"answer": answer})

    # Multiple candidates
    options = [{
        "number": int(i+1),
        "question": str(faq_data[idx]["question"]),
        "index": int(idx)
    } for i, (idx, _) in enumerate(candidates)]

    session["pending_options"] = options
    options_text = "\n".join([f"{opt['number']}. {opt['question']}" for opt in options])
    response_text = f"Multiple matches found. Please select by number:\n{options_text}"
    
    conversation_history.append({"question": user_query, "answer": response_text})
    session["conversation_history"] = conversation_history
    
    return jsonify({
        "answer": response_text,
        "options": [opt["question"] for opt in options]
    })

@app.route("/reset", methods=["POST"])
def reset():
    session.pop("conversation_history", None)
    session.pop("pending_options", None)
    return jsonify({"message": "Conversation history reset."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)