import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os

# Configuration
GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "greetings", "Salam"]
DEFAULT_RESPONSE = "I'm sorry, I don't have an answer to that question yet. Your question has been saved so I can learn and improve. Please feel free to ask another question!"
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

CORS(app, supports_credentials=True, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

def save_unanswered_question(question):
    file_path = "unanswered_questions.json"
    data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:  
                    data = json.loads(content)
        except (json.JSONDecodeError, IOError):
            data = []
    if question not in data:
        data.append(question)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


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

# Debug endpoint to check session
@app.route("/debug", methods=["GET"])
def debug():
    session_id = request.cookies.get('session')
    return jsonify({
        "session_id": session_id,
        "session_data": dict(session),
        "has_pending_options": "pending_options" in session,
        "pending_options_count": len(session.get("pending_options", [])),
        "conversation_history_count": len(session.get("conversation_history", []))
    })

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query", "").strip().lower()
    
    # Debug: Print session info
    print(f"=== DEBUG INFO ===")
    print(f"Query: {user_query}")
    print(f"Session ID: {request.cookies.get('session')}")
    print(f"Session keys: {list(session.keys())}")
    print(f"Has pending_options: {'pending_options' in session}")
    if "pending_options" in session:
        print(f"Pending options count: {len(session['pending_options'])}")
    print(f"==================")

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
            print(f"Debug: Selected number: {selected_num}, Available options: {len(pending_options)}")
            if 1 <= selected_num <= len(pending_options):
                selected_idx = pending_options[selected_num - 1]["index"]
                print(f"Debug: Selected index: {selected_idx}")
                answer = faq_data[selected_idx]["answer"]
                conversation_history.append({"question": f"Selected option {selected_num}", "answer": answer})
                session["conversation_history"] = conversation_history
                session.pop("pending_options")
                return jsonify({"answer": answer})
            else:
                session.pop("pending_options")
                return jsonify({"answer": "Invalid selection. Please try again."})
        except ValueError:
            # If it's not a number, treat it as a new query
            session.pop("pending_options")
            print(f"Debug: Not a number, treating as new query: {user_query}")

    # Context handling
    if conversation_history and ("that" in user_query or "more" in user_query):
        last_topic = conversation_history[-1]["question"]
        user_query = f"{last_topic} {user_query}"

    # Process query
    candidates = find_best_answers(user_query)
    
    if not candidates:
        print(f"No match found for: {user_query}")
        save_unanswered_question(user_query)
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
    session["conversation_history"] = conversation_history
    print(f"Debug: Stored {len(options)} options in session")
    
    options_text = "<br>".join([f"{opt['number']}. {opt['question']}" for opt in options])
    response_text = f"Multiple matches found. Please select by number:<br>{options_text}"
    
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