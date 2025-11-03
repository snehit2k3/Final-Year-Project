from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Initialize Flask App
app = Flask(__name__)
# Enable CORS to allow your React app to make requests to this API
CORS(app)

# --- Load Trained Model and Vectorizer ---
try:
    model = joblib.load('reentrancy_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model or vectorizer files not found. Please run train.py first.")
    model = None
    vectorizer = None

# --- Define the Analysis Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze_contract():
    if not model or not vectorizer:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    data = request.get_json()
    if not data or 'source_code' not in data:
        return jsonify({'error': 'Missing "source_code" in request body.'}), 400

    source_code = data['source_code']
    if not isinstance(source_code, str) or len(source_code.strip()) == 0:
        return jsonify({'error': ' "source_code" must be a non-empty string.'}), 400

    # --- Prediction Logic ---
    code_vector = vectorizer.transform([source_code])
    prediction = model.predict(code_vector)[0]
    probability = model.predict_proba(code_vector)[0]
    
    # --- NEW: Stricter Confidence Threshold ---
    # We only flag as vulnerable if the model predicts reentrancy (class 1) AND
    # its confidence in that prediction is very high (e.g., > 98%).
    # This reduces false positives on safe contracts. You can tune this value.
    CONFIDENCE_THRESHOLD = 0.98 
    is_vulnerable = bool(prediction == 1 and probability[1] > CONFIDENCE_THRESHOLD)

    # --- Generate Report ---
    # Use the probability of the predicted class for the report
    report_confidence = probability[prediction] * 100

    if is_vulnerable:
        report = {
            "vulnerable": True,
            "vulnerability": "Reentrancy",
            "status": "❌ Found",
            "confidence": f"{report_confidence:.2f}%",
            "problem": "The model detected patterns consistent with reentrancy vulnerabilities. This often involves an external call (like `call.value()`) being made before a state variable (like a balance) is updated.",
            "risk": "An attacker could potentially call a function repeatedly before its first invocation completes, allowing them to drain funds or manipulate contract state.",
            "fix": "Apply the 'Checks-Effects-Interactions' pattern. Perform all checks and state changes (e.g., updating balances) *before* making external calls."
        }
    else:
        # This branch is now taken if the contract is predicted as clean OR
        # if it's predicted as vulnerable but with low confidence.
        report = {
            "vulnerable": False,
            "status": "✅ Not Detected",
            "confidence": f"{report_confidence:.2f}%",
            "summary": "The contract does not show common textual patterns associated with reentrancy bugs, or the confidence was below the required threshold.",
        }

    return jsonify(report)

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

