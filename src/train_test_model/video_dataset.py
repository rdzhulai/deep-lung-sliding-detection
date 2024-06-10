import os
import torch
from torch.utils.data import Dataset
from config.config import DATA_DIR_PATH
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as F

class VideoDataset(Dataset):
    def __init__(self, df, datadir, transform=None):
        self.df = df
        self.datadir = datadir
        self.data = df["video_name"].tolist()
        self.targets = torch.tensor(df["video_class"].tolist(), dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_info = self.df.iloc[idx]
        video_name = video_info['video_name']
        start_frame = video_info['start']
        end_frame = video_info['end']

        # Read and crop images in batches
        frames = []
        for frame_idx in range(start_frame, end_frame):
            frame_path = os.path.join(self.datadir, video_name, f"frame_{frame_idx}.png")
            frame = read_image(frame_path, ImageReadMode.GRAY).to(torch.float32)
            # Apply transformations
            frame = self.transform(frame)
            frames.append(frame)

        # Convert frames to tensor
        frames = torch.stack(frames)

        return frames, self.targets[idx]
