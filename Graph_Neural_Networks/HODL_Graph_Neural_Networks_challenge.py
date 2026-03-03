import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from sklearn.metrics import f1_score
from copy import deepcopy
import math
from torch.utils.data import Dataset
from typing import Any

device = "cuda" if torch.cuda.is_available() else "cpu"

class MazeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MazeConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # MLP for message computation: concatenate x_i and x_j
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels)
        )
        
        # MLP for update step
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels)
        )
        
        # Skip connection layer
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index):
        # Store input for skip connection
        x_input = x
        
        # Propagate messages
        out = self.propagate(edge_index, x=x)
        
        # Update with skip connection
        x = self.update_mlp(torch.cat([x_input, out], dim=-1))
        x = x + self.skip(x_input)  # Skip connection
        
        return x

    def message(self, x_i, x_j):
        # Concatenate node features with neighbor features
        message_input = torch.cat([x_i, x_j], dim=-1)
        return self.message_mlp(message_input)


class MazeGNN(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=8):
        super().__init__()
        self.dropout = 0.2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder and decoder
        self.encoder = self.get_mlp(2, 32, hidden_dim)
        self.decoder = self.get_mlp(hidden_dim, 64, 2, last_relu=False)

        # Create multiple convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(MazeConv(hidden_dim, hidden_dim))
        
        # Input preservation layer and MLP for concatenation
        self.input_preserve = nn.Linear(2, hidden_dim)
        self.concat_mlp = self.get_mlp(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, data, num_nodes):
        x, edge_index = data.x, data.edge_index
        input_features = x.clone()
        
        # Encode node features
        x = self.encoder(x)
        
        # Add input preservation
        input_encoded = self.input_preserve(input_features)
        x = x + input_encoded  # Preserve original input information
        
        # Dynamic number of layers based on graph size
        effective_layers = min(self.num_layers, max(4, int(math.log2(num_nodes))))
        
        # Graph computations with multiple layers
        for i in range(effective_layers):
            # Before each convolution (except first), concatenate with input and put through MLP
            if i > 0:
                x_with_input = torch.cat([x, input_encoded], dim=-1)
                x = self.concat_mlp(x_with_input)
            
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final decoding
        x = self.decoder(x)
        
        return F.log_softmax(x, dim=1)

    def get_mlp(self, input_dim, hidden_dim, output_dim, last_relu=True):
        modules = [
            nn.Linear(input_dim, int(hidden_dim)), 
            nn.ReLU(), 
            nn.Dropout(self.dropout), 
            nn.Linear(int(hidden_dim), output_dim)
        ]
        if last_relu:
            modules.append(nn.ReLU())
        return nn.Sequential(*modules)


def init_model() -> torch.nn.Module:
    # Use more layers and larger hidden dimension for better generalization
    model = MazeGNN(hidden_dim=64, num_layers=12)
    return model.to(device)


def train_model(model: torch.nn.Module, train_dataset: Dataset[Any]) -> torch.nn.Module:
    epochs = 60
    lr = 0.0004
    
    # Use provided train_dataset
    dataset = train_dataset
    
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Split dataset for validation
    val_split = 0.2
    dataset_size = len(dataset)
    train_size = int((1 - val_split) * dataset_size)
    
    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, range(train_size))
    val_subset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))
    
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1)

    model.train()
    worst_loss = -1
    best_model = None

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)

            pred = model(data, data.num_nodes)
            loss = criterion(pred, data.y.to(torch.long))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data, data.num_nodes)
                loss = criterion(pred, data.y.to(torch.long))
                val_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Get evaluation metrics
        eval_result = eval_model(model, val_loader)
        graph_acc_str = eval_result.split("graph accuracy: ")[-1]
        graph_val = float(graph_acc_str)
        
        print(f'Epoch: {epoch + 1} train_loss: {avg_train_loss:.5f} val_loss: {avg_val_loss:.5f} \t {eval_result}')
        
        comp = (avg_val_loss, -graph_val)
        if worst_loss == -1 or comp < worst_loss:
            worst_loss = comp
            best_model = deepcopy(model)
            print(f"New best model - val_loss: {avg_val_loss:.5f}, graph_acc: {graph_val:.3f}")

    return best_model


def eval_model(model, dataset, mode=None):
    model.eval()
    acc = 0
    tot_nodes = 0
    tot_graphs = 0
    perf = 0
    gpred = []
    gsol = []

    for step, batch in enumerate(dataset):
        n = len(batch.x) / batch.num_graphs
        
        # Apply size filtering based on mode
        if mode == "small" and n > 4*4:
            continue
        elif mode == "medium" and n > 8*8:
            continue
        elif mode == "large" and n > 16*16:
            continue
        
        with torch.no_grad():
            batch = batch.to(device)
            pred = model(batch, int(n))
        
        y_pred = torch.argmax(pred, dim=1)
        tot_nodes += len(batch.x)
        tot_graphs += batch.num_graphs
        
        graph_acc = torch.sum(y_pred == batch.y).item()
        acc += graph_acc
        
        gpred.extend(y_pred.cpu().numpy())
        gsol.extend(batch.y.cpu().numpy())
        
        if graph_acc == n:
            perf += 1

    if len(gpred) == 0:
        return "node accuracy: 0.000 | node f1 score: 0.000 | graph accuracy: 0.000"
    
    f1score = f1_score(gsol, gpred, zero_division=0)
    
    node_accuracy = acc / tot_nodes if tot_nodes > 0 else 0
    graph_accuracy = perf / tot_graphs if tot_graphs > 0 else 0
    
    return f"node accuracy: {node_accuracy:.3f} | node f1 score: {f1score:.3f} | graph accuracy: {graph_accuracy:.3f}"