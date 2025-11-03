import pandas as pd
import torch
from torch_geometric.data import Data
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

def contract_to_graph(source_code: str, label: int, vectorizer: TfidfVectorizer):
    """
    Parses a Solidity source code string, extracts its CFG, and converts it into a
    PyTorch Geometric Data object with rich TF-IDF node features.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False, encoding='utf-8') as tmp:
        tmp.write(source_code)
        tmp_path = tmp.name

    try:
        solc_version = get_solc_version(source_code)
        slither_instance = Slither(tmp_path, solc=SOLC_PATH, solc_solcs_select=solc_version)
        
        all_nodes, node_map, node_expressions = [], {}, []

        for contract in slither_instance.contracts:
            for function in contract.functions:
                if not function.nodes: continue
                for node in function.nodes:
                    if node not in node_map:
                        node_map[node] = len(all_nodes)
                        all_nodes.append(node)
                        node_expressions.append(str(node.expression) if node.expression else "")

        if not all_nodes:
            os.remove(tmp_path)
            return None

        all_edges = []
        for contract in slither_instance.contracts:
            for function in contract.functions:
                if not function.nodes: continue
                for node in function.nodes:
                    if hasattr(node, 'successors'):
                        for successor in node.successors:
                            if successor in node_map:
                                all_edges.append((node_map[node], node_map[successor]))
        
        if node_expressions:
            node_features = torch.tensor(vectorizer.transform(node_expressions).toarray(), dtype=torch.float)
        else:
            os.remove(tmp_path)
            return None
            
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous() if all_edges else torch.empty((2, 0), dtype=torch.long)
        graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
        
        os.remove(tmp_path)
        return graph_data

    except Exception as e:
        os.remove(tmp_path)
        return None

def main():
    if not SOLC_PATH:
        print("Error: 'solc' executable not found.")
        return

    print("Loading dataset from CSV...")
    df = pd.read_csv('data/dataset.csv')
    df = df[df['label'].isin(['reentrancy', 'clean'])]
    df['label_binary'] = df['label'].apply(lambda x: 1 if x == 'reentrancy' else 0)
    
    print("Building vocabulary from all node expressions...")
    all_expressions = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Pass 1: Building Vocab"):
        source_code = str(row['source_code'])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False, encoding='utf-8') as tmp:
            tmp.write(source_code)
            tmp_path = tmp.name
        try:
            solc_version = get_solc_version(source_code)
            slither_instance = Slither(tmp_path, solc=SOLC_PATH, solc_solcs_select=solc_version)
            for contract in slither_instance.contracts:
                for function in contract.functions:
                    if not function.nodes: continue
                    for node in function.nodes:
                        all_expressions.append(str(node.expression) if node.expression else "")
        except Exception:
            pass
        finally:
            os.remove(tmp_path)

    print(f"Vocabulary built from {len(all_expressions)} node expressions.")
    vectorizer = TfidfVectorizer(max_features=128)
    vectorizer.fit(all_expressions)
    
    # --- Save the fitted vectorizer for the GNN API ---
    joblib.dump(vectorizer, 'gnn_vectorizer.pkl')
    print("✅ GNN feature vectorizer saved to gnn_vectorizer.pkl")
    
    print("Starting graph conversion with rich features...")
    graph_dataset = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Pass 2: Creating Graphs"):
        graph = contract_to_graph(str(row['source_code']), row['label_binary'], vectorizer)
        if graph:
            graph_dataset.append(graph)

    print(f"\nSuccessfully converted {len(graph_dataset)} contracts into graphs.")
    output_path = "data/graph_dataset.pt"
    torch.save(graph_dataset, output_path)
    print(f"✅ Graph dataset saved to {output_path}")

if __name__ == "__main__":
    main()

