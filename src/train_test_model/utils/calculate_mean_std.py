import torch

def calculate_overall_mean_of_means_and_stds(data_loader, device):
    channel_means = []
    channel_stds = []
    
    for batch_frames, _ in data_loader:
        batch_frames = batch_frames.to(device)
        
        # Calculate mean and std across all pixels for each channel
        mean = torch.mean(batch_frames, dim=(0, 2, 3))
        std = torch.std(batch_frames, dim=(0, 2, 3))
        
        channel_means.append(mean)
        channel_stds.append(std)

    # Calculate the overall mean and std across all batches
    all_channel_means = torch.stack(channel_means).mean(dim=0)
    all_channel_stds = torch.stack(channel_stds).mean(dim=0)

    # Calculate the mean of all_channel_means and all_channel_stds
    mean_of_means = torch.mean(all_channel_means)
    mean_of_stds = torch.mean(all_channel_stds)

    return [mean_of_means.item()], [mean_of_stds.item()]
