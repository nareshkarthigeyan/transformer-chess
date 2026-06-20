import torch
import torch.nn as nn

def train_epoch(model, dataloader_or_data, criterion, optimizer, device):
    """Runs a single training epoch."""
    model.train()
    X, Y = dataloader_or_data
    X, Y = X.to(device), Y.to(device)
    
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()