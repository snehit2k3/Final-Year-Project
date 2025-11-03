import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib

# Using an advanced Graph Attention Network (GATv2) Model Definition
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_heads=4):
        super(GAT, self).__init__()
        torch.manual_seed(42)
        
        # A deep, 3-layer GNN architecture for learning complex patterns
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=num_heads)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv3 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads)

        # A more powerful classification head with an additional layer
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

def main():
    print("Loading full graph dataset...")
    try:
        dataset = torch.load("data/graph_dataset.pt", weights_only=False)
    except FileNotFoundError:
        print("Error: data/graph_dataset.pt not found.")
        print("Please run the full 'create_graph_dataset.py' script first.")
        return

    random.shuffle(dataset)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Dataset loaded. Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    all_labels = [data.y.item() for data in train_dataset]
    class_counts = np.bincount(all_labels)
    class_weights = torch.tensor([1.0, float(class_counts[0]) / class_counts[1]], dtype=torch.float)
    print(f"Calculated class weights: {class_weights}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_features = dataset[0].num_node_features
    model = GAT(num_node_features=num_features, hidden_channels=32) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("\nStarting final advanced GAT-GNN training...")
    for epoch in range(1, 71): # Train for 70 epochs for robust learning
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)

        print(f'Epoch: {epoch:02d}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    print("\nTraining finished. Evaluating model on the test set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.tolist())
            all_labels.extend(data.y.tolist())

    print("\n--- Final GNN Model Performance (Advanced) ---")
    print(classification_report(all_labels, all_preds, target_names=['Clean', 'Reentrancy'], zero_division=0))

    # --- Save the trained GNN model ---
    model_save_path = "reentrancy_gnn_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Trained GNN model saved to {model_save_path}")

if __name__ == "__main__":
    main()

