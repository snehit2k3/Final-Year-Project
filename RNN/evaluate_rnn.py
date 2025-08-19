import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("reentrancy_lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load test data
# Replace this with your actual test data loading
# Example:
# X_test_raw = ["contract1 code ...", "contract2 code ..."]
# y_test = [1, 0]

# 👉 Replace with your actual test dataset (as lists)
import pandas as pd
df = pd.read_csv("PrimeSmartVuln.csv")  # Example file
X_test_raw = df["source_code"].tolist()
y_test = df["label"].tolist()

# Preprocess
max_length = 512
X_test_seq = tokenizer.texts_to_sequences(X_test_raw)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Predict
y_pred_probs = model.predict(X_test_pad)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n================== Evaluation Report ==================")
print("✅ Accuracy :", round(acc, 4))
print("📌 Precision:", round(prec, 4))
print("📌 Recall   :", round(rec, 4))
print("📌 F1 Score :", round(f1, 4))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Safe", "Vulnerable"]))
print("========================================================")
