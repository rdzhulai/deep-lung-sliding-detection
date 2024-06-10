import torch
import click
from config.config import device, DATA_DIR_PATH, df, MODELS_DIR_PATH
import models
from focal_loss import FocalLoss
from video_dataset import VideoDataset
from utils.checkpoint_functions import save_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loops import test_loop
from utils.print_history import print_history
from utils.prepare_data import prepare_data
import torchvision.transforms.v2 as T
import sys

@click.command()
@click.argument('checkpoint_name')
def main(checkpoint_name):
    # Load the checkpoint
    checkpoint = torch.load(f"{MODELS_DIR_PATH}/{checkpoint_name}")

    hyperparameters = checkpoint['hyperparameters']
        
    # Prepare test data
    *_, test_df, _ = prepare_data(
        df,
        hyperparameters['sampled_size'], 
        hyperparameters['train_test_size'], 
        hyperparameters['val_test_size'], 
        hyperparameters['seed'],
        hyperparameters['class_ratio']
    )
    
    # Check if testing has already been done
    if checkpoint['history']['test']['loss'][-1]:
        print_history(checkpoint['history'], 'test', epoch=-1)
        sys.exit(0)
    
    # Define test data transformations
    test_transform = T.Compose([
        T.Resize(size=hyperparameters['size'], interpolation=T.InterpolationMode.BILINEAR),
        T.Normalize(torch.Tensor(hyperparameters['mean_std'][0]), torch.Tensor(hyperparameters['mean_std'][1])),
    ])
    
    # Create test dataset
    test_dataset = VideoDataset(test_df, DATA_DIR_PATH, test_transform)

    # Create test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'])

    # Load the model
    model = getattr(models, hyperparameters['model_name'])(
        (hyperparameters['batch_size'], 1, hyperparameters['frames'], hyperparameters['size'][0], hyperparameters['size'][1]),
        hyperparameters['dropout_prob']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    # Define the loss criterion
    criterion = FocalLoss(hyperparameters['alpha'], hyperparameters['gamma'], "mean")

    # Perform testing
    with tqdm(test_dataloader, desc="Testing") as tqdm_test_dataloader:
        test_loop(model, criterion, tqdm_test_dataloader, checkpoint['history']['test'], device)

    # Print test history
    print_history(checkpoint['history'], 'test', epoch=-1)

    # Save the checkpoint
    torch.save(checkpoint, f"{MODELS_DIR_PATH}/{checkpoint_name}")
    print(f"Checkpoint was saved with name {checkpoint_name}")

if __name__ == "__main__":
    main()
