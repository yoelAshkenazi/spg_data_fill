import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from fp_builds import make_graph as mg
# Step 1: Load and preprocess the Banknote dataset
from sklearn.datasets import fetch_openml

# Load Banknote dataset from OpenML
banknote = fetch_openml(name="banknote-authentication", version=1)
X = banknote.data
y = banknote.target.astype(int)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
x = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y.values)

# make edges.
distances = mg.calc_l2(pd.DataFrame(X))
edges = mg.get_knn_edges(distances, 5)

print(edges.shape)
edge_index = edges
# # Step 2: Construct a k-NN graph
# # You can use a higher value for k if your dataset is large
# k = 6
# transform = KNNGraph(k=k)
# print('before')
# edge_index = transform(Data(x=x)).edge_index
# edge_index = to_undirected(edge_index)  # Ensure undirected edges


# Step 3: Create Data object
data = Data(x=x, edge_index=edge_index, y=y)

# Step 4: Split data into train and validation sets
train_idx, val_idx = train_test_split(range(data.num_nodes), test_size=0.2, stratify=data.y)

# Create train and val masks
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
data.train_mask = train_mask
data.val_mask = val_mask

# Step 5: Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, _data):
        x, edge_index = _data.x, _data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and loss function
model = GCN(in_channels=data.num_features, hidden_channels=16, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 6: Train the GCN Model
def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    out = out[data.train_mask].squeeze()
    true_y = data.y[data.train_mask]
    print(out.shape, true_y.shape)
    print(f'out: {out.shape}, true_y: {true_y.shape}')
    loss = F.cross_entropy(out, true_y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Step 7: Evaluate the model
def evaluate(data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
    return train_acc.item(), val_acc.item()

# Training loop
for epoch in range(1, 201):
    loss = train(data)
    train_acc, val_acc = evaluate(data)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Final Evaluation
train_acc, val_acc = evaluate(data)
print(f"Final Train Acc: {train_acc:.4f}, Final Val Acc: {val_acc:.4f}")