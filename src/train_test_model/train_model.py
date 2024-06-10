import click
import torch
from config.config import device, DATA_DIR_PATH, df, hyperparameters, MODELS_DIR_PATH
import models
from focal_loss import FocalLoss
from video_dataset import VideoDataset
from utils.checkpoint_functions import save_checkpoint
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loops import train_loop, val_loop
from utils.print_history import print_history
from utils.prepare_data import prepare_data 
from utils.find_batch_size import get_batch_size 
from utils.calculate_mean_std import calculate_overall_mean_of_means_and_stds 
import sys
import torchvision.transforms.v2 as T
from random import randint

@click.command()
@click.option("--new_checkpoint_name", "-n", default="model", help="Name of the checkpoint")
@click.option("--load_checkpoint_name", "-l", default=None, help="Path to model weights")
@click.option("--epochs", "-e", help="How many epochs to train", type=int)
def main(new_checkpoint_name, load_checkpoint_name, epochs): 
    # Check if epochs provided
    if epochs is None:
        print("Please provide the number of epochs.")
        sys.exit(1)

    # Initialize history
    history = {
            "train": {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "specificity": [],
            "confusion_matrix": [],
            "time": []
            }, 
            "val": {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "specificity": [],
            "confusion_matrix": [],
            "time": []
            }, 
            "test": {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "specificity": [],
            "confusion_matrix": [],
            "time": []
            } 
        }
    
    start_epoch = 0
    checkpoint = None
    if load_checkpoint_name:
        print("Loading weights from:", load_checkpoint_name)
        checkpoint = torch.load(f"{MODELS_DIR_PATH}/{load_checkpoint_name}")
        hyperparameters.update(checkpoint['hyperparameters'])
        history.update(checkpoint['history'])

        model = getattr(models, hyperparameters['model_name'])(
            (hyperparameters['batch_size'], 1, hyperparameters['frames'], hyperparameters['size'][0], hyperparameters['size'][1]),
            hyperparameters['dropout_prob']
        ).to(device)
        optimizer = Adam(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs += checkpoint['epoch']
        start_epoch = checkpoint['epoch']
    
    if hyperparameters['seed'] == "auto":
        hyperparameters['seed'] = randint(0, 999999)
        print(f"seed={hyperparameters['seed']}")
        
    torch.manual_seed(hyperparameters['seed'])
    
    if hyperparameters['class_ratio'] == "auto":
        train_df, val_df, test_df, class_ratio = prepare_data(
            df,
            hyperparameters['sampled_size'], 
            hyperparameters['train_test_size'], 
            hyperparameters['val_test_size'], 
            hyperparameters['seed']
        )
        hyperparameters['class_ratio'] = class_ratio
        print(f"class_ratio={hyperparameters['class_ratio']}")
    else:
        train_df, val_df, test_df, _ = prepare_data(
            df,
            hyperparameters['sampled_size'], 
            hyperparameters['train_test_size'], 
            hyperparameters['val_test_size'], 
            hyperparameters['seed'],
            hyperparameters['class_ratio']
        )
        
        
    if load_checkpoint_name is None:
        if hyperparameters['batch_size'] == "auto":
            model = getattr(models, hyperparameters['model_name'])(
                (2, 1, hyperparameters['frames'], hyperparameters['size'][0], hyperparameters['size'][1]),
                hyperparameters['dropout_prob']
            )
            optimizer = Adam(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
            criterion = FocalLoss(alpha=1, gamma=0, reduction="mean")
            
            hyperparameters['batch_size'] = get_batch_size(
                model,
                criterion,
                optimizer,
                (hyperparameters['frames'], 1, hyperparameters['size'][0], hyperparameters['size'][1]),
                (1,),
                len(train_df),
                device
            )
            del model, optimizer
            torch.cuda.empty_cache()
            print(f"batch_size={hyperparameters['batch_size']}")
            
        assert all(len(df) % hyperparameters['batch_size']  != 1 for df in [train_df, val_df, test_df] if df is not None), "The length of the data is not compatible with batch normalization due to the batch size."
                
        model = getattr(models, hyperparameters['model_name'])(
            (hyperparameters['batch_size'], 1, hyperparameters['frames'], hyperparameters['size'][0], hyperparameters['size'][1]),
            hyperparameters['dropout_prob']
        ).to(device)
        optimizer = Adam(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])

        if hyperparameters['sched_epochs'] == "auto":
            hyperparameters['sched_epochs'] = epochs
            print(f"sched_epochs={hyperparameters['sched_epochs']}")
    
    if hyperparameters['alpha'] == "auto":
        hyperparameters['alpha'] = len(train_df[train_df['video_class'] == 0]) / len(train_df)
        print(f"alpha={hyperparameters['alpha']}")
        
    criterion = FocalLoss(alpha=hyperparameters['alpha'], gamma=hyperparameters['gamma'], reduction="mean")
    
    if hyperparameters['mean_std'] == "auto":
        train_transform = T.Compose([
            T.Resize(size=hyperparameters['size'], interpolation=T.InterpolationMode.BILINEAR),
        ])
        train_dataset = VideoDataset(train_df, DATA_DIR_PATH, train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'])
        hyperparameters['mean_std'] = calculate_overall_mean_of_means_and_stds(train_dataloader, device)
        print(f"mean_std={hyperparameters['mean_std']}")
    
    train_transform = T.Compose([
        T.Resize(size=hyperparameters['size'], interpolation=T.InterpolationMode.BILINEAR),
        T.RandomApply([
            T.RandomHorizontalFlip(),
        ], p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=(0, 0.2), contrast=(0, 0.2), saturation=(0, 0.2)),
        ], p=0.5),
        T.RandomApply([
            T.RandomRotation(degrees=(-10, 10)),
        ], p=0.5),
        T.RandomApply([
            T.RandomResizedCrop(size=hyperparameters['size'], scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        ], p=0.5),
        T.Normalize(mean=hyperparameters['mean_std'][0], std=hyperparameters['mean_std'][1]),
    ])

    train_dataset = VideoDataset(train_df, DATA_DIR_PATH, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    
    
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, hyperparameters['learning_rate'], epochs=hyperparameters['sched_epochs'], steps_per_epoch=len(train_dataloader))
        
    if checkpoint:
        sched.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    
    if(val_df is not None):
        val_transform = T.Compose([
            T.Resize(size=hyperparameters['size'], interpolation=T.InterpolationMode.BILINEAR),
            T.Normalize(torch.Tensor(hyperparameters['mean_std'][0]), torch.Tensor(hyperparameters['mean_std'][1])),
        ])
        val_dataset = VideoDataset(val_df, DATA_DIR_PATH, val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters['batch_size'])


    for epoch in range(start_epoch, epochs):
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training") as tqdm_train_dataloader:
            
            if epoch >= hyperparameters['sched_epochs']:
                train_loop(model, criterion, optimizer, tqdm_train_dataloader, history['train'], device, grad_clip=hyperparameters['grad_clip'])
            else:
                train_loop(model, criterion, optimizer, tqdm_train_dataloader, history['train'], device, lr_scheduler=sched, grad_clip=hyperparameters['grad_clip'])
        
        print_history(history, 'train', epoch)
        
        if(val_df is not None):
            with tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation") as  tqdm_val_dataloader:
                val_loop(model, criterion, tqdm_val_dataloader, history['val'], device)

            print_history(history,'val',  epoch)
            
        for k in history['test'].keys():
            history['test'][k].append(None)

        save_checkpoint(model, criterion, optimizer, sched, epoch+1, history, hyperparameters, new_checkpoint_name, MODELS_DIR_PATH)

if __name__ == "__main__":
    main()
