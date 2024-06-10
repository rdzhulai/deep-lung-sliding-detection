from torchvision.ops.focal_loss import sigmoid_focal_loss  # Importing necessary function
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = 'none'):
        super(FocalLoss, self).__init__()  # Calling the constructor of the parent class (nn.Module)
        self.alpha = alpha  # Initializing alpha parameter
        self.gamma = gamma  # Initializing gamma parameter
        self.reduction = reduction  # Initializing reduction method

    def forward(self, inputs, targets):
        # Using the imported sigmoid_focal_loss function with the provided parameters
        return sigmoid_focal_loss(inputs=inputs, targets=targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
