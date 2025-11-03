from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from slither import Slither
import tempfile
import os
import re
import shutil

# --- 1. Define the GNN Model Architecture ---
# This MUST be the exact same architecture as the one in your final train_gnn.py
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_heads=4):
        super(GAT, self).__init__()
        torch.manual_seed(42)
        
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=num_heads)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv3 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads)

        self.lin1 = torch.nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        self.lin2 = torch.nn.Linear(hidden_channels * num_heads, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

# --- 2. Load the Trained Model and Vectorizer ---
print("Loading GNN model and vectorizer...")
try:
    vectorizer = joblib.load('gnn_vectorizer.pkl')
    print("✅ Vectorizer loaded successfully.")
    
    # Instantiate the model with the correct dimensions based on the vectorizer
    # Using the more compatible 'vocabulary_' attribute to get feature count
    NUM_NODE_FEATURES = len(vectorizer.vocabulary_)
    model = GAT(num_node_features=NUM_NODE_FEATURES, hidden_channels=32)
    
    # Load the trained weights (the "brain")
    model.load_state_dict(torch.load('reentrancy_gnn_model.pth'))
    model.eval() # Set to evaluation mode
    print("✅ GNN model loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    print("Please ensure 'reentrancy_gnn_model.pth' and 'gnn_vectorizer.pkl' are present.")
    model = None
    vectorizer = None

# --- 3. Helper functions for parsing new contracts ---
SOLC_PATH = shutil.which("solc")

def get_solc_version(source_code: str):
    """
    Extracts a usable solc version from the pragma line.
    Handles simple versions, '^', and '>=' pragmas.
    """
    # Look for a pragma line
    pragma_match = re.search(r"pragma solidity\s*([^;]+);", source_code)
    if not pragma_match:
        return None
    
    version_str = pragma_match.group(1)
    
    # Find any version number (e.g., 0.8.20)
    # This regex is improved to handle common version specifiers
    version_match = re.search(r"(\d+\.\d+\.\d+)", version_str)
    if version_match:
        return version_match.group(1)
        
    return None

def contract_to_graph_for_prediction(source_code: str, vectorizer: joblib.memory.MemorizedResult):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False, encoding='utf-8') as tmp:
        tmp.write(source_code)
        tmp_path = tmp.name
    
    try:
        solc_version = get_solc_version(source_code)
        slither_instance = Slither(tmp_path, solc=SOLC_PATH, solc_solcs_select=solc_version)
        
        node_map, node_expressions, all_edges = {}, [], []

        for contract in slither_instance.contracts:
            for function in contract.functions:
                if not function.nodes: continue
                for node in function.nodes:
                    if node not in node_map:
                        node_map[node] = len(node_expressions)
                        node_expressions.append(str(node.expression) if node.expression else "")
                
                for node in function.nodes:
                    if hasattr(node, 'successors'):
                        for successor in node.successors:
                            if successor in node_map:
                                all_edges.append((node_map[node], node_map[successor]))
        
        os.remove(tmp_path)

        if not node_expressions: return None
        
        node_features = torch.tensor(vectorizer.transform(node_expressions).toarray(), dtype=torch.float)
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous() if all_edges else torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=node_features, edge_index=edge_index)

    except Exception as e:
        # Add detailed logging to the server console for debugging
        print(f"--- SLITHER PARSING FAILED ---")
        print(f"Error: {e}")
        print(f"Attempted solc version: {solc_version}")
        print(f"------------------------------")
        os.remove(tmp_path)
        return None

# --- 4. Initialize Flask App and Configure CORS ---
app = Flask(__name__)
# This simple configuration is robust for development and allows all origins.
CORS(app)

# --- 5. Define the Analysis Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze_contract():
    if not model or not vectorizer or not SOLC_PATH:
        error_msg = "Server is not ready. Model, vectorizer, or solc compiler is missing."
        return jsonify({'error': error_msg}), 500

    data = request.get_json()
    if not data or 'source_code' not in data:
        return jsonify({'error': 'Missing "source_code" in request body.'}), 400

    source_code = data['source_code']
    
    graph_data = contract_to_graph_for_prediction(source_code, vectorizer)
    
    if graph_data is None:
        return jsonify({'error': 'Failed to parse the smart contract. It may contain syntax errors or an unsupported pragma version.'}), 400

    data_batch = Batch.from_data_list([graph_data])

    with torch.no_grad():
        out = model(data_batch.x, data_batch.edge_index, data_batch.batch)
        probabilities = F.softmax(out, dim=1)[0]
        prediction = probabilities.argmax().item()

    is_vulnerable = bool(prediction == 1)
    confidence = probabilities[prediction].item() * 100

    if is_vulnerable:
        report = {
            "vulnerable": True, "vulnerability": "Reentrancy", "status": "❌ Found",
            "confidence": f"{confidence:.2f}%",
            "problem": "The GNN model detected a structural pattern consistent with reentrancy vulnerabilities.",
            "risk": "An external call appears to be made before a critical state variable is updated.",
            "fix": "Apply the 'Checks-Effects-Interactions' pattern. Ensure all state changes are made before external calls."
        }
    else:
        report = {
            "vulnerable": False, "status": "✅ Not Detected", "confidence": f"{confidence:.2f}%",
            "summary": "The contract's control flow graph does not show patterns typically associated with reentrancy bugs."
        }
        
    return jsonify(report)

# --- 6. Run the App ---
if __name__ == '__main__':
    # Run on port 5002 to avoid conflict with the text-based model server
    app.run(host='0.0.0.0', port=5002, debug=True)

