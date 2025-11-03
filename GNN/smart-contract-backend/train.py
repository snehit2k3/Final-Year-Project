import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression # <--- REMOVED
import lightgbm as lgb # <--- ADDED: Import LightGBM
from sklearn.metrics import classification_report
import joblib
import re

print("Starting model training process...")

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv('data/dataset.csv')
    SOURCE_CODE_COLUMN = 'source_code'
    LABEL_COLUMN = 'label'
except FileNotFoundError:
    print("Error: Dataset file not found. Make sure 'your_dataset.csv' is in the 'data' folder.")
    exit()

df = df[df[LABEL_COLUMN].isin(['reentrancy', 'clean'])]
print(f"Loaded {len(df)} relevant contracts.")

df.dropna(subset=[SOURCE_CODE_COLUMN], inplace=True)
df[SOURCE_CODE_COLUMN] = df[SOURCE_CODE_COLUMN].astype(str)
df['label_binary'] = df[LABEL_COLUMN].apply(lambda x: 1 if x == 'reentrancy' else 0)

# --- 2. Feature Extraction (TF-IDF) ---
print("Extracting features using TF-IDF...")
# <--- CHANGED: Updated vectorizer parameters to capture more detail
vectorizer = TfidfVectorizer(
    min_df=3,           # Ignore terms that appear in less than 3 contracts
    max_features=10000, # Use the top 10,000 most relevant terms
    ngram_range=(1, 3)  # Consider single words, two-word, and three-word phrases
)

X = vectorizer.fit_transform(df[SOURCE_CODE_COLUMN])
y = df['label_binary']

# --- 3. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- 4. Model Training ---
# <--- CHANGED: Swapped Logistic Regression for the more powerful LightGBM model
print("Training LightGBM model...")
model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- 5. Model Evaluation ---
print("Evaluating model performance...")
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Clean', 'Reentrancy']))

# --- 6. Save the Model and Vectorizer ---
print("Saving model and vectorizer to files...")
joblib.dump(model, 'reentrancy_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\n✅ Model training complete. 'reentrancy_model.pkl' and 'tfidf_vectorizer.pkl' are saved.")