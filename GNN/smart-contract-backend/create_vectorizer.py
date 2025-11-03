import pandas as pd
from slither import Slither
import tempfile
import os
from tqdm import tqdm
import re
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# This forces the script to find the 'solc' executable managed by solc-select
SOLC_PATH = shutil.which("solc")

def get_solc_version(source_code: str):
    """
    Extracts the solidity version from the pragma line using regex.
    """
    pragma_match = re.search(r"pragma solidity\s*([^;]+);", source_code)
    if not pragma_match: return None
    version_str = pragma_match.group(1)
    version_match = re.search(r"\d+\.\d+\.\d+", version_str)
    if not version_match: return None
    return version_match.group(0)

def main():
    if not SOLC_PATH:
        print("Error: 'solc' executable not found.")
        return

    print("Loading dataset from CSV...")
    df = pd.read_csv('data/dataset.csv')
    df = df[df['label'].isin(['reentrancy', 'clean'])]
    
    # --- This script ONLY builds the vocabulary and saves the vectorizer ---
    print("Building vocabulary from all node expressions (this is the only pass)...")
    all_expressions = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Building Vocab"):
        source_code = str(row['source_code'])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False, encoding='utf-8') as tmp:
            tmp.write(source_code)
            tmp_path = tmp.name
        try:
            solc_version = get_solc_version(source_code)
            slither_instance = Slither(tmp_path, solc=SOLC_PATH, solc_solcs_select=solc_version)
            # --- FIX: Iterate through each contract, then get the functions for THAT contract ---
            for contract in slither_instance.contracts:
                for function in contract.functions: # This was the line with the bug
                    if not function.nodes: continue
                    for node in function.nodes:
                        all_expressions.append(str(node.expression) if node.expression else "")
        except Exception as e:
            # print(f"Skipping contract due to error: {e}") # Uncomment for debugging
            pass
        finally:
            os.remove(tmp_path)

    if not all_expressions:
        print("Error: No node expressions were extracted. Cannot create a vectorizer.")
        return

    print(f"Vocabulary built from {len(all_expressions)} node expressions.")
    vectorizer = TfidfVectorizer(max_features=128) # Must match the features in train_gnn.py
    vectorizer.fit(all_expressions)
    
    # Save the fitted vectorizer for the GNN API
    vectorizer_path = 'gnn_vectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n✅ GNN feature vectorizer saved to {vectorizer_path}")
    print("You can now proceed with training the GNN model.")

if __name__ == "__main__":
    main()

