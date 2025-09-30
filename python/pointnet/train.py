from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import PointNetConv
from pointnet.dataset import PointNetDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

lr = 0.05

# Define a simple 2-layer GATv2
# GNN-not working
local_nn = nn.Sequential(
    nn.Linear(6, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 256),
    nn.BatchNorm1d(256),
)

global_nn = nn.Sequential(
    nn.Linear(256, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 5),  # predict first 5 eigenvalues
)

def train(model, optimizer, train_data, val_data, device, writer):
    # Print model info
    print(model)
    N_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.size())
    print(f"Total number of trainable parameters: {N_params}")

    loss_fn = nn.BCEWithLogitsLoss()

    global_step = 0
    # Training loop
    for epoch in range(200):
        model.train()
        total_loss = 0
        for data in train_data:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data.x, data.pos, data.edge_index)
            label = data.y[:, :5]
            loss = loss_fn(out, label)
            # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            writer.add_scalar('Train/Step_Loss', loss.item(), global_step)
            global_step += 1

        # Validation
        eval_loss = evaluate(model, val_data, device)
        avg_train_loss = total_loss / len(train_data)
        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d} | Loss {avg_train_loss:.4f} | Val Loss {eval_loss:.4f}")

        writer.add_scalar('Train/Epoch_Loss', avg_train_loss, epoch)
        writer.add_scalar('Val/Epoch_Loss', eval_loss, epoch)

def evaluate(model, dataset, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    eval_loss = 0
    for data in dataset:
        data = data.to(device)
        out = model(data.x, data.pos, data.edge_index)
        label = data.y[:, :5]
        eval_loss += loss_fn(out, label).item()
    eval_loss /= len(dataset)
    return eval_loss

if __name__ == "__main__":
    # get and split dataset
    root = Path("~/Documents/phy/python/data/chair_600").expanduser()
    dataset = PointNetDataset(root)
    N = len(dataset)
    N_train = int(N * 0.8)
    N_val = int(N * 0.1)
    N_test = N - N_train - N_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [N_train, N_val, N_test],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )
    
    # log part
    log_dir = Path("~/Documents/phy/python/pointnet/logs/pointnet2").expanduser()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(log_dir)+time_str)
    
    cuda_index = 2
    device = torch.device(f"cuda:{cuda_index}")

    # start with simple task, say first eigenvalue's 5 regions
    model = PointNetConv(local_nn, global_nn).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    train(model, optimizer, train_dataset, val_dataset, device, writer)
    eval_loss = evaluate(model, test_dataset, device)
    print(f"Test Loss {eval_loss:.4f}")
    
    # Save the trained model
    model_path = Path("~/Documents/phy/python/pointnet/outputs/pointnet2_model.pth").expanduser()
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    writer.close()