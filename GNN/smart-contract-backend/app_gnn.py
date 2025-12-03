import re
import os
import shutil
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Helper functions for parsing new contracts ---
# We define these globally, but they will rely on resources loaded lazily
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

# --- GLOBAL VARIABLES (LAZY LOADING) ---
model = None
vectorizer = None
torch = None
Slither = None
Data = None
Batch = None

# We need to define GAT inside the loader or make it accessible
GAT = None

def load_resources():
    """
    Loads PyTorch, Slither, and the Models only when needed.
    This prevents Render 'Timed Out' errors during startup.
    """
    global model, vectorizer, torch, Slither, Data, Batch, GAT
    
    if model is None:
        print("⚡ Loading GNN model, libraries, and vectorizer... (First run only)")
        try:
            # 1. Import heavy libraries here
            import joblib
            import torch as t
            import torch.nn.functional as F
            from torch_geometric.nn import GATv2Conv, global_mean_pool
            from torch_geometric.data import Data as D, Batch as B
            from slither import Slither as S
            
            # Assign to globals so other functions can use them
            torch = t
            Slither = S
            Data = D
            Batch = B

            # 2. Define the GNN Model Architecture (Must match training)
            class GAT_Model(torch.nn.Module):
                def __init__(self, num_node_features, hidden_channels, num_heads=4):
                    super(GAT_Model, self).__init__()
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
            
            GAT = GAT_Model

            # 3. Load the Vectorizer
            if os.path.exists('gnn_vectorizer.pkl'):
                vectorizer = joblib.load('gnn_vectorizer.pkl')
                print("✅ Vectorizer loaded successfully.")
            else:
                raise FileNotFoundError("'gnn_vectorizer.pkl' not found.")

            # 4. Instantiate and Load Model
            NUM_NODE_FEATURES = len(vectorizer.vocabulary_)
            model = GAT(num_node_features=NUM_NODE_FEATURES, hidden_channels=32)
            
            if os.path.exists('reentrancy_gnn_model.pth'):
                model.load_state_dict(torch.load('reentrancy_gnn_model.pth'))
                model.eval() # Set to evaluation mode
                print("✅ GNN model loaded successfully.")
            else:
                raise FileNotFoundError("'reentrancy_gnn_model.pth' not found.")

        except Exception as e:
            print(f"❌ Error loading GNN resources: {e}")
            raise e

def contract_to_graph_for_prediction(source_code: str):
    # Ensure resources are loaded before running this
    if not Slither or not vectorizer:
        raise RuntimeError("Resources not loaded. Call load_resources() first.")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False, encoding='utf-8') as tmp:
        tmp.write(source_code)
        tmp_path = tmp.name
    
    try:
        solc_version = get_solc_version(source_code)
        # Use the global Slither class we imported lazily
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
        
        # Use global torch
        node_features = torch.tensor(vectorizer.transform(node_expressions).toarray(), dtype=torch.float)
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous() if all_edges else torch.empty((2, 0), dtype=torch.long)
        
        # Use global Data
        return Data(x=node_features, edge_index=edge_index)

    except Exception as e:
        print(f"--- SLITHER PARSING FAILED ---")
        print(f"Error: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None

# --- Initialize Flask App and Configure CORS ---
app = Flask(__name__)
CORS(app)

# --- Define the Analysis Endpoint ---
@app.route('/', methods=['GET'])
def home():
    return "GNN Backend is Running! Send POST requests to /analyze."

@app.route('/analyze', methods=['POST'])
def analyze_contract():
    try:
        # 1. Trigger Lazy Loading
        load_resources()
        
        if not model or not vectorizer:
            return jsonify({'error': 'Server initialization failed. Model or vectorizer missing.'}), 500
        
        if not SOLC_PATH:
             return jsonify({'error': 'Server error: solc compiler not found.'}), 500

        data = request.get_json()
        if not data or 'source_code' not in data:
            return jsonify({'error': 'Missing "source_code" in request body.'}), 400

        source_code = data['source_code']
        
        # 2. Parse Contract
        graph_data = contract_to_graph_for_prediction(source_code)
        
        if graph_data is None:
            return jsonify({'error': 'Failed to parse the smart contract. It may contain syntax errors or an unsupported pragma version.'}), 400

        # 3. Predict
        data_batch = Batch.from_data_list([graph_data])

        import torch.nn.functional as F # Ensure F is available if not global
        
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

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server for GNN model...")
    # Run on port 5002 to avoid conflict with the text-based model server
    app.run(host='0.0.0.0', port=5002, debug=True)