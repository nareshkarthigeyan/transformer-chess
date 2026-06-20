import torch
import os

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)
    
    for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Real-time tracking so you aren't left guessing
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"  -> Batch [{batch_idx + 1}/{total_batches}] | Current Loss: {loss.item():.4f}")
        
    return running_loss / total_batches

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f" Saved checkpoint to {path}")