import torch
import pandas as pd
import json

# Load hyperparameters from JSON file
with open("./config/hyperparameters.json", "r") as file:
    hyperparameters = json.load(file)

# Define paths for data and saved models
DATA_DIR_PATH = "../../data"
MODELS_DIR_PATH = "../saved_models"

# Determine device (CPU or GPU) for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set cudnn benchmark for faster training on CUDA
torch.backends.cudnn.benchmark = True

# Load sliding metafile JSON data
with open('./config/sliding_metafile.json', 'r') as file:
    data = json.load(file)

# Extract video clips based on sliding window intervals
clips = []

for row in data:
    for n in range(row['cut_interval'][0][0], row['cut_interval'][0][1] - hyperparameters['frames'] + 1, hyperparameters['frames']):
        new_el = {"video_name": row['video_name'], "start": n, "end": n + hyperparameters['frames'], "video_class": row['video_class']}
        clips.append(new_el)

# Create a DataFrame containing the extracted video clips
df = pd.DataFrame(clips)
