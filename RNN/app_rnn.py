import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS # Imported from the GNN server reference

# --- Helper functions from your original script ---

# We assume this file exists in the same directory
try:
    from reentrancy_rule_checker import detect_external_before_state_update
except ImportError:
    print("Error: Could not import 'detect_external_before_state_update' from 'reentrancy_rule_checker.py'.")
    print("Please make sure the file exists and the function is correctly named.")
    # Create a dummy function to allow the server to run, but it will warn the user.
    def detect_external_before_state_update(code):
        print("Warning: Rule checker not loaded.")
        return False

# Parameters
MAX_LENGTH = 512

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
    clean_code = clean_solidity_code(code)
    seq = tokenizer.texts_to_sequences([clean_code])
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

# --- Flask Web Server Setup ---

print("Loading RNN model and tokenizer...")
# Load model and tokenizer (globally, once)
try:
    model = load_model("reentrancy_lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✅ RNN Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or tokenizer: {e}")
    print("Please make sure 'reentrancy_lstm_model.h5' and 'tokenizer.pkl' are in the same directory.")
    model = None
    tokenizer = None

# Initialize the Flask app and configure CORS
app = Flask(__name__)
CORS(app) # Added from the GNN server reference

@app.route('/analyze', methods=['POST'])
def analyze_contract_rnn():
    """API endpoint to predict reentrancy using the RNN model."""
    if not model or not tokenizer:
        return jsonify({'error': 'Server is not ready. RNN Model or tokenizer is missing.'}), 500

    try:
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
        return jsonify({"error": "Internal Server Error"}), 500

# Run the app
if __name__ == "__main__":
    print("Starting Flask server for RNN model on http://127.0.0.1:5001")
    # Use debug=True as per GNN server reference
    app.run(port=5001, debug=True, host='0.0.0.0')

