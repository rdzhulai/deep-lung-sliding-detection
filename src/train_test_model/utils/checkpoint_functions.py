import os
import torch
from datetime import datetime

def save_checkpoint(model, criterion, optimizer,  lr_scheduler, epoch, history, hyperparameters, indentifier, path):
    model_name = f"{indentifier}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'history': history,
            'hyperparameters': hyperparameters,
            'epoch': epoch
        }, os.path.join(path, model_name))
    print(f"Checkpoint was saved with name {model_name}")