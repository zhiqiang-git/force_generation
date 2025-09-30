from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from gnn.dataset import TetMeshDataset
from torch.utils.tensorboard import SummaryWriter

lr = 0.01
y_min = 0.0
y_max = 100.0

# Define a simple 2-layer GATv2
# GNN-not working
class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6)
        self.conv3 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        x = torch.sigmoid(x)
        return x

def train(model, optimizer, train_data, val_data, device, writer):
    # Print model info
    print(model)
    N_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.size())
    print(f"Total number of trainable parameters: {N_params}")

    global_step = 0
    # Training loop
    for epoch in range(200):
        model.train()
        total_loss = 0
        for data in train_data:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data.x, data.edge_index)
            label = (data.y - y_min) / (y_max - y_min)
            loss = F.mse_loss(out, label[:, :5])
            # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            writer.add_scalar('Train/Step_Loss', loss.item(), global_step)
            global_step += 1

        # Validation
        eval_loss = evaluate(model, val_data, device)
        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val Loss {eval_loss:.4f}")
        
        avg_train_loss = total_loss / len(train_data)
        writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)
        writer.add_scalar('Val/Epoch_Loss', eval_loss, epoch)

def evaluate(model, dataset, device):
    model.eval()
    eval_loss = 0
    for data in dataset:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        label = (data.y - y_min) / (y_max - y_min)
        eval_loss += F.mse_loss(out, label[:, :5]).item()
    eval_loss /= len(dataset)
    return eval_loss

if __name__ == "__main__":
    # get and split dataset
    root = Path("~/Documents/phy/python/data/chair_600").expanduser()
    dataset = TetMeshDataset(root)
    N = len(dataset)
    N_train = int(N * 0.8)
    N_val = int(N * 0.1)
    N_test = N - N_train - N_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [N_train, N_val, N_test],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )
    
    # log part
    log_dir = Path("~/Documents/phy/python/gnn/logs/gatv2").expanduser()
    writer = SummaryWriter(log_dir=str(log_dir))
    
    cuda_index = 1
    device = torch.device(f"cuda:{cuda_index}")

    # start with simple task, say first eigenvalue's 5 regions
    model = GATv2(dataset.num_features, 64, 5).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    train(model, optimizer, train_dataset, val_dataset, device, writer)
    eval_loss = evaluate(model, test_dataset, device)
    print(f"Test Loss {eval_loss:.4f}")
    
    # Save the trained model
    model_path = Path("~/Documents/phy/python/gnn/outputs/gatv2_model.pth").expanduser()
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    writer.close()