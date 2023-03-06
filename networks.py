import torch
import torch.nn as nn
from layers import *

# NOTE:
# - ReconstructionNetwork, CapsuleNetwork are specific, they are designed for the MNIST dataset.
# - DNE: Does Not Exist
# - Masking does expand the computational graph, it is not detached.
# - n = batch size

class ReconstructionNetwork(nn.Module):
    def __init__(self, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        self.fc1 = nn.Linear(in_features=10*16, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=1*28*28)
        
        self.to(device)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x.view(-1, 1, 28, 28)

class CapsuleNetwork(nn.Module):
    def __init__(self, threshold=0.5, alpha=0.0005, lamb=0.5, m_minus=0.1, m_plus=0.9, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=256, 
                                kernel_size=(9,9), stride=1, padding='valid', bias=True, 
                                device=device, dtype=dtype)
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=32*8, 
                                kernel_size=(9,9), stride=2, padding='valid', bias=True, 
                                device=device, dtype=dtype)
        self.c2c_3 = ConvToCapLayer(6*6, 32, 8, device, dtype)
        self.caps_4 = CapsuleLayer(10, 6*6, 32, 16, 8, 3, device, dtype)
        self.mask_5 = MaskLayer(10, threshold, flatten=True)
        self.rec_net_6 = ReconstructionNetwork(device, dtype)
        self.cost_7 = CapsuleNetworkCostLayer(alpha, lamb, m_minus, m_plus, device, dtype)
        
        self.to(self.device)
    
    def forward(self, x, y=None):
        # x:input, y:label
        # x shape: (n or DNE, c1, h1, w1)
        # y.shape = (n or DNE, 10) or None
        
        x = self.conv_1(x)
        # x shape: (n or DNE, c2, h2, w2)
        x = self.conv_2(x)
        # x shape: (n or DNE, c3, h3, w3)
        x = self.c2c_3(x)
        # x shape: (n, 10, 6*6*32, 8, 1)
        x = self.caps_4(x)
        # x shape: (n, 10, 16)
        reconstructions = self.mask_5(x, y)
        # reconstructions.shape = (n, 160)
        reconstructions = self.rec_net_6(reconstructions)
        # shape: (n, 1, 28, 28)
        
        # capsule predictions, reconstructions
        # shape: (n, 10, 16), (n, 1, 28, 28)
        return x, reconstructions

    def cost(self, input_data, labels, reconstructions, capsule_predictions):
        return self.cost_7(input_data, labels, reconstructions, capsule_predictions)

class Evaluator(object):
    def __init__(self) -> None:
        pass

    def accuracy(self, capsule_network_outputs, labels):
        # capsule_network_outputs shape: (n, n_capsules, n_capsule_features)
        # labels shape: (n or DNE, n_classes)
        # n_classes = n_capsules
        
        labels = labels.view(-1, labels.shape[-1])
        # labels shape: (n, n_classes)
        
        capsule_network_outputs = torch.norm(capsule_network_outputs, dim=2)
        # capsule_network_outputs shape: (n, n_classes)
        
        accuracy = torch.round(labels-capsule_network_outputs)
        # accuracy shape: (n, n_classes)
        
        accuracy = torch.any(accuracy, dim=1)
        # accuracy shape: (n)
        
        accuracy = 1 - torch.sum(accuracy)/accuracy.shape[0]
        # accuracy shape: (1,)
        
        return round(accuracy.item(), 2)
