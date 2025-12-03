import re
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Helper functions from your original script ---

# We assume this file exists in the same directory
try:
    from reentrancy_rule_checker import detect_external_before_state_update
except ImportError:
    print("Warning: Could not import 'detect_external_before_state_update'. Using dummy function.")
    def detect_external_before_state_update(code):
        return False

# Initialize the Flask app
app = Flask(__name__)

# --- CORS FIX ---
# Allow all origins ("*") to fix the immediate access issue.
# We also explicitly allow the OPTIONS method which browsers use for preflight checks.
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"])

# --- GLOBAL VARIABLES (LAZY LOADING) ---
model = None
tokenizer = None
pad_sequences = None # We will load this function dynamically too

# Parameters
MAX_LENGTH = 512

def load_resources():
    """
    Loads TensorFlow model and Tokenizer only when needed.
    This prevents Render 'Timed Out' errors during startup.
    """
    global model, tokenizer, pad_sequences
    
    if model is None:
        print("⚡ Loading RNN model and tokenizer... (First run only)")
        try:
            # Import heavy libraries here, not at the top
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences as ps
            import pickle
            
            # Set the global pad_sequences function
            pad_sequences = ps

            # Load Model
            model = load_model("reentrancy_lstm_model.h5")
            
            # Load Tokenizer
            with open("tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
                
            print("✅ RNN Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

# Clean Solidity code
def clean_solidity_code(code):
    """Removes comments, pragmas, and extra whitespace from Solidity code."""
    code = re.sub(r"//.*", "", code)  # Remove single-line comments
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)  # Remove multi-line comments
    code = re.sub(r"pragma solidity.*?;", "", code)  # Remove pragma lines
    code = re.sub(r"\s+", " ", code).strip()  # Normalize whitespace
    return code

# Predict using RNN model
def predict_reentrancy(code):
    """Runs the RNN model prediction on the cleaned code."""
    # Ensure resources are loaded
    load_resources()
    
    clean_code = clean_solidity_code(code)
    seq = tokenizer.texts_to_sequences([clean_code])
    # Use the globally loaded pad_sequences function
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    prob = model.predict(padded, verbose=0)[0][0]
    return float(prob) # Ensure it's a JSON-serializable float

# Combine rule-based + model decision
def final_classification(code):
    """Combines RNN and rule-based checks for a final verdict."""
    rnn_prob = predict_reentrancy(code)
    rule_flag = detect_external_before_state_update(code)

    # Final decision
    if rnn_prob > 0.5 or rule_flag:
        label = "⚠️ Likely Vulnerable"
    else:
        label = "✅ Likely Safe"

    return label, rnn_prob, rule_flag

@app.route('/', methods=['GET'])
def home():
    return "RNN Backend is Running! Send POST requests to /predict."

# --- ROUTE FIX: Changed from /analyze to /predict to match frontend ---
@app.route('/predict', methods=['POST'])
def analyze_contract_rnn():
    """API endpoint to predict reentrancy using the RNN model."""
    try:
        # 1. Trigger Lazy Loading
        load_resources()
        
        data = request.get_json()
        
        # Use 'source_code' key to match the GNN server
        if not data or 'source_code' not in data:
            return jsonify({'error': 'Missing "source_code" in request body.'}), 400

        solidity_code = data['source_code']
        
        # Run the same classification logic
        verdict, rnn_prob, rule_flag = final_classification(solidity_code)

        # --- Create detailed JSON report (inspired by GNN server) ---
        if verdict == "⚠️ Likely Vulnerable":
            report = {
                "vulnerable": True,
                "status": "⚠️ Likely Vulnerable",
                "model_type": "RNN (Text-Based) + Rules",
                "confidence_rnn": f"{rnn_prob * 100:.2f}%",
                "rule_triggered": bool(rule_flag),
                "problem": "The RNN model detected sequential patterns associated with reentrancy, or a rule-based check was triggered.",
                "risk": "There is a high probability that an external call is made before a state update, or a known unsafe pattern exists.",
                "fix": "Apply the 'Checks-Effects-Interactions' pattern. Ensure all state changes (e.g., updating balances) are made *before* external calls (e.g., msg.sender.call.value())."
            }
        else: # "✅ Likely Safe"
            report = {
                "vulnerable": False,
                "status": "✅ Likely Safe",
                "model_type": "RNN (Text-Based) + Rules",
                "confidence_rnn": f"{rnn_prob * 100:.2f}%",
                "rule_triggered": bool(rule_flag),
                "summary": "The RNN model did not detect vulnerable sequential patterns, and the rule-based check passed."
            }
            
        return jsonify(report)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    print("Starting Flask server for RNN model...")
    # Use environment variable for PORT if available (Best practice for Render)
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)