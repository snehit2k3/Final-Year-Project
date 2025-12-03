import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import random
import numpy as np

# --- Configuration ---
MODEL_PATH = 'reentrancy_gnn_model.pth'
DATA_PATH = "data/graph_dataset.pt"
# These labels match your goal screenshot
CLASS_NAMES = ["Safe", "Vulnerable"] 
# 1 = Vulnerable (Reentrancy)
POSITIVE_CLASS_LABEL = 1 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 
# --- MODEL DEFINITION ---
# This is your exact GAT model class from train_gnn.py
#
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

def load_test_data():
    """
    Loads the full dataset and recreates the *exact* test split
    used during training, thanks to random_state=42.
    """
    print(f"Loading full graph dataset from {DATA_PATH}...")
    try:
        # weights_only=False is correct as it loads Data objects
        dataset = torch.load(DATA_PATH, weights_only=False) 
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        print("Please ensure the graph dataset exists.")
        return None, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This may be due to a PyTorch version mismatch or file corruption.")
        return None, None

    random.shuffle(dataset)
    # This split is identical to your training script
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    
    print(f"Dataset loaded. Using test set size: {len(test_dataset)}")
    
    # We use shuffle=False for evaluation
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get the number of features from the first data object
    num_features = dataset[0].num_node_features
    
    return test_loader, num_features

def evaluate_model():
    """
    Loads the trained GNN model, runs evaluation, and prints
    the report in your desired format.
    """
    
    ### 1. Load Data ###
    test_loader, num_features = load_test_data()
    if test_loader is None:
        return

    ### 2. Initialize Model ###
    print(f"Loading model from {MODEL_PATH}...")
    # We use the same parameters as your training script
    model = GAT(num_node_features=num_features, hidden_channels=32)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("This likely means the model definition here does not match")
        print("the model that was saved. (This shouldn't happen now)")
        return
        
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode

    all_labels = []
    all_preds = []

    ### 3. Run Predictions ###
    print("Running predictions on test data...")
    with torch.no_grad(): # Disable gradient calculations
        for data in test_loader:
            data = data.to(DEVICE)
            
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1) # Get the predicted class (0 or 1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    if not all_labels:
        print("Error: No labels found. Did the test loader work?")
        return

    ### 4. Calculate and Print Metrics ###
    # This is the format from your *first* screenshot
    print("\n" + "="*30)
    print("======= Evaluation Report =======")
    print("="*30)

    accuracy = accuracy_score(all_labels, all_preds)
    # Note: We specify pos_label to get the metrics for the "Vulnerable" class
    precision = precision_score(all_labels, all_preds, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)

    print(f"✅ Accuracy : {accuracy:.4f}")
    print(f"📍 Precision: {precision:.4f}  (For {CLASS_NAMES[POSITIVE_CLASS_LABEL]} class)")
    print(f"📍 Recall   : {recall:.4f}  (For {CLASS_NAMES[POSITIVE_CLASS_LABEL]} class)")
    print(f"📍 F1 Score : {f1:.4f}  (For {CLASS_NAMES[POSITIVE_CLASS_LABEL]} class)")

    # This is the detailed report
    print("\n📊 Classification Report:\n")
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=CLASS_NAMES, # Using "Safe" and "Vulnerable"
        zero_division=0
    )
    print(report)


if __name__ == "__main__":
    evaluate_model()