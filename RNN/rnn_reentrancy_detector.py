# rnn_reentrancy_detector.py

import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
from reentrancy_rule_checker import check_external_before_state_update

### Step 1: Load & Filter Data
df = pd.read_csv("PrimeSmartVuln.csv")

# Filter bad/missing data
df = df.dropna(subset=["source_code", "reentrancy"])
df = df[df["source_code"].str.len() > 100]

# Label: 1 if reentrancy > 0
df["label"] = df["reentrancy"].apply(lambda x: 1 if int(x) > 0 else 0)

### Step 2: Clean Solidity Code
def clean_solidity_code(code):
    code = re.sub(r"//.*", "", code)  # single-line comments
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)  # multi-line comments
    code = re.sub(r"pragma solidity.*?;", "", code)  # version pragma
    code = re.sub(r"\s+", " ", code).strip()  # normalize whitespace
    return code

df["source_code"] = df["source_code"].apply(clean_solidity_code)

### Step 3: Split Dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

### Step 4: Tokenization
vocab_size = 10000
max_length = 512
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["source_code"])

def tokenize_and_pad(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

X_train = tokenize_and_pad(train_df["source_code"])
X_val = tokenize_and_pad(val_df["source_code"])
X_test = tokenize_and_pad(test_df["source_code"])

y_train = train_df["label"].values
y_val = val_df["label"].values
y_test = test_df["label"].values

### Step 5: LSTM Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

### Step 6: Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

### Step 7: Evaluate
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_classes))

### Step 8: Save Model + Tokenizer
model.save("reentrancy_lstm_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
