import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torchvision.transforms.functional as F
import torchvision.transforms as T

class OpticalFlowExtractor(nn.Module):
    def __init__(self):
        super(OpticalFlowExtractor, self).__init__()
        # Initialize the RAFT optical flow model
        raft_weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=raft_weights, progress=False)
        # Define the transformation
        self.transform = raft_weights.transforms()
        self.flow_step = 1

    def forward(self, batch_frames):
        # Expand batch_frames to 3 channels
        batch_frames = torch.stack([batch_frames[i].expand(-1, 3, -1, -1) for i in range(batch_frames.size(0))])
        batch_flow_list = []
        # Loop through each frame in the batch
        for i in range(batch_frames.size(0)):
            # Resize frames and apply transformation
            batch1 = F.resize(batch_frames[i, :-self.flow_step], size=[256, 256], antialias=False)
            batch2 = F.resize(batch_frames[i, self.flow_step:], size=[256, 256], antialias=False)
            batch1, batch2 = self.transform(batch1, batch2)

            # Compute optical flow
            flow = self.model(batch1, batch2)[-1]
            batch_flow_list.append(flow)

        return torch.stack(batch_flow_list)

class CMWRA(nn.Module):
    def __init__(self, input_shape, dropout_prob=0.5):
        super(CMWRA, self).__init__()
        
        # Initialize the OpticalFlowExtractor
        self.optical_flow_extractor = OpticalFlowExtractor()
        
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(2, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            nn.Conv3d(16, 32, kernel_size=(2, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        )

        # Calculate the size of the fully connected input
        with torch.no_grad():
            dummy_input = torch.zeros(input_shape[0], 2, input_shape[2]-1, 256, 256)
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.view(conv_output.size(0), -1).size(1)

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # Perform optical flow extraction
        x = self.optical_flow_extractor(x).permute(0, 2, 1, 3, 4)
        # Pass through convolutional layers
        x = self.conv_layers(x)
        return self.fc_layers(x)

class BasicBlock3D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout_prob=0.0):
        super(BasicBlock3D, self).__init__()
        # Define 3D convolutional layers
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        # Optional downsample layer
        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes)
            )
        self.dropout = nn.Dropout3d(p=dropout_prob)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
class CMWRU(nn.Module):
    def __init__(self, input_shape, dropout_prob=0.5):
        super(CMWRU, self).__init__()
        
        self.input_shape = input_shape
        self.in_planes = 8
        
        # Define the convolutional layers
        self.conv1 = nn.Conv3d(self.input_shape[1], 8, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2), bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Define the residual layers
        self.layer1 = nn.Sequential(
            BasicBlock3D(self.in_planes, 16, stride=1, dropout_prob=dropout_prob),
            BasicBlock3D(16, 16, stride=1, dropout_prob=dropout_prob)
        )
        self.in_planes = 16
        self.layer2 = nn.Sequential(
            BasicBlock3D(self.in_planes, 32, stride=2, dropout_prob=dropout_prob),
            BasicBlock3D(32, 32, stride=1, dropout_prob=dropout_prob)
        )
        self.in_planes = 32
        self.layer3 = nn.Sequential(
            BasicBlock3D(self.in_planes, 64, stride=2, dropout_prob=dropout_prob),
            BasicBlock3D(64, 64, stride=1, dropout_prob=dropout_prob)
        )

        # Calculate the size of the fully connected input
        with torch.no_grad():
            dummy_input = torch.zeros(self.input_shape)
            conv_output = self._forward_conv(dummy_input)
            conv_output_size = conv_output.view(conv_output.size(0), -1).size(1)

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(32, 1)
        )

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x.permute(0, 2, 1, 3, 4))
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
class Resblock(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        # Define the residual block with 2 convolutional layers
        self.conv1 = nn.Conv2d(in_channels=shape, out_channels= shape, kernel_size= 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=shape, out_channels= shape, kernel_size= 3, padding=1)
        self.relul = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relul(out)
        out = self.conv2(out)
        out = self.relul(out)

        return out+x
    
class RCM2d_Small(nn.Module):
    def __init__(self, input_shape, dropout_prob=0.5):
        super(RCM2d_Small, self).__init__()
        
        self.input_shape = input_shape
        self.hidden_units = 256
        self.output_shape = 1
        self.num_layers = 1

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Conv2d(in_channels=self.input_shape[1], out_channels=4, kernel_size= 7, padding=3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            Resblock(4),
            nn.MaxPool2d(4),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size= 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            Resblock(8),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size= 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            Resblock(16),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(self.hidden_units*16,self.hidden_units*2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units*2)
        )
        
        conv_output_size = self.hidden_units*2
        
        # Define the GRU layer
        self.gru = nn.GRU(conv_output_size, self.hidden_units*4, self.num_layers, batch_first=True)

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.hidden_units*4,self.output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = torch.stack([self.conv_layers(x[i]) for i in range(x.size(0))])
        out, _ = self.gru(x)
        out = self.fc_layers(out[:,-1,:])
        return out