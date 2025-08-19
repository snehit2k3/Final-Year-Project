# inference_rnn.py

import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from reentrancy_rule_checker import check_external_before_state_update

# Load model and tokenizer
model = load_model("reentrancy_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Parameters
MAX_LENGTH = 512

# Clean Solidity code
def clean_solidity_code(code):
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    code = re.sub(r"pragma solidity.*?;", "", code)
    code = re.sub(r"\s+", " ", code).strip()
    return code

# Predict using RNN model
def predict_reentrancy(code):
    clean_code = clean_solidity_code(code)
    seq = tokenizer.texts_to_sequences([clean_code])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    prob = model.predict(padded, verbose=0)[0][0]
    return prob

# Combine rule-based + model decision
def final_classification(code):
    rnn_prob = predict_reentrancy(code)
    rule_flag = check_external_before_state_update(code)

    # Final decision
    if rnn_prob > 0.5 or rule_flag:
        label = "⚠️ Likely Vulnerable"
    else:
        label = "✅ Likely Safe"

    return label, rnn_prob, rule_flag

# Run inference on an example Solidity contract
if __name__ == "__main__":
    with open("example_contract.sol", "r") as f:
        solidity_code = f.read()

    verdict, confidence, rule_detected = final_classification(solidity_code)

    print("\n================ Reentrancy Detection Report ================")
    print("🔍 Final Verdict               :", verdict)
    print("🧠 RNN Vulnerability Probability:", f"{confidence:.4f}")
    print("🧾 Rule-based Red Flag Detected:", rule_detected)
    print("============================================================\n")
