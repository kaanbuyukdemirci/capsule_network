import torch
import torch.nn as nn

# NOTE:
# - CapsuleLayer, ConvToCapLayer, and CapsuleNetworkCostLayer are general, they can be used by other networks.
# - DNE: Does Not Exist
# - n = batch size

class CapsuleLayer(nn.Module):
    """A capsule layer with support for weight sharing. Supported input shapes are:
        (batch_size or DNE, n_in_capsules_unique, n_in_capsules_sharing_per_unique, n_in_features, 1 or DNE)
        (batch_size or DNE, n_in_capsules, n_in_features, 1 or DNE)
    and the output shape:
        (batch_size, n_out_capsules, n_out_features)
    where 
        n_in_capsules = n_in_capsules_unique * n_in_capsules_sharing_per_unique
    
    'n_in_capsules_unique' and 'n_in_capsules_sharing_per_unique' are for weight sharing. For the architecture 
    mentioned in the original paper "Dynamic Routing Between Capsules", these values would be:
        n_in_capsules_unique = 6*6
        n_in_capsules_sharing_per_unique = 32
    
    and the other parameters would be:
        n_out_capsules = 10
        n_out_features = 16
        n_in_capsules = 6*6*32
        n_in_features = 8
        
    For no weight sharing, you can set the parameters as:
        n_in_capsules_unique = 6*6*32
        n_in_capsules_sharing_per_unique = 1
    
    see the example in this repo.
    """
    def __init__(self, n_out_capsules, n_in_capsules_unique, n_in_capsules_sharing_per_unique, n_out_features, n_in_features, n_iterations=3, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        self.n_in_capsules_unique = n_in_capsules_unique
        self.n_in_capsules_sharing_per_unique = n_in_capsules_sharing_per_unique
        self.n_in_capsules = n_in_capsules_unique * n_in_capsules_sharing_per_unique
        self.n_in_features = n_in_features
        self.n_out_capsules = n_out_capsules
        self.n_out_features = n_out_features
        self.n_iterations = n_iterations
        
        self.weights = nn.Parameter(torch.randn(size=(1, n_out_capsules, n_in_capsules_unique, 1, n_out_features, n_in_features), dtype=self.dtype) * 
                                    torch.sqrt(torch.tensor(2/(self.n_in_capsules*n_in_features + n_out_capsules*n_out_features))),
                                    requires_grad=True) # N(0, 2/(fan_in+fan_out))
        # self.biases = nn.Parameter(torch.randn(size=(1, n_out_capsules, 1, n_out_features, 1), dtype=self.dtype) * 
        #                             torch.sqrt(torch.tensor(2/(n_out_capsules*n_out_features))),
        #                             requires_grad=True)
        # u.shape = (batch_size, 1             , n_in_capsules_unique, n_in_capsules_sharing_per_unique, n_in_features , 1            )
        # W.shape = (1         , n_out_capsules, n_in_capsules_unique, 1                               , n_out_features, n_in_features)
        # b.shape = (1,        , n_out_capsules, 1                                                     , n_out_features, 1)
        # see that capsules in 3rd dimension share weights, while capsules in 2nd dimension have unique weights.
        # batch_size, n_out_capsules, 1, n_out_features, 1
        
        self.to(device)
    
    @staticmethod
    def squash(x, dim):
        # input shape: (..., b, ...)
        norm = torch.linalg.norm(x, dim=dim, keepdim=True)
        # norm shape: (..., 1, ...)
        # return shape: (..., b, ...)
        return x * norm / (1 + norm**2)
    
    def forward(self, x):
        # supported input shapes: 
        # (batch_size or DNE, n_in_capsules_unique, n_in_capsules_sharing_per_unique, n_in_features, 1 or DNE)
        # (batch_size or DNE, n_in_capsules, n_in_features, 1 or DNE)
        
        x = x.view(-1, 1, self.n_in_capsules_unique, self.n_in_capsules_sharing_per_unique, self.n_in_features, 1)
        # x = u
        # u.shape = (batch_size, 1, n_in_capsules_unique, n_in_capsules_sharing_per_unique, n_in_features, 1)
        
        x = torch.matmul(self.weights, x)
        # x = u_hat
        # u_hat.shape = (batch_size, n_out_capsules, n_in_capsules_unique, n_in_capsules_sharing_per_unique, n_out_features, 1)
        
        x = x.view(-1, self.n_out_capsules, self.n_in_capsules, self.n_out_features, 1)
        # x = u_hat
        # u_hat.shape = (batch_size, n_out_capsules, n_in_capsules, n_out_features, 1)
        
        u_hat = x.detach()
        # this will be needed later in b updates. We are detaching so while we are calculating c iteratively, 
        # we will not be expanding the computational graph.
        
        b = torch.zeros(size=(x.shape[0], self.n_out_capsules, self.n_in_capsules, 1, 1), dtype=self.dtype, device=self.device, requires_grad=False)
        # b.shape = (batch_size, n_out_capsules, n_in_capsules, 1, 1)
        
        c = torch.softmax(b, dim=1)
        # c.shape = (batch_size, n_out_capsules, n_in_capsules, 1, 1)
        
        v = x.detach()
        # v = x
        # this will be needed later in b updates. We are detaching so while we are calculating c iteratively, 
        # we will not be expanding the computational graph.
        
        for _ in range(self.n_iterations):
            v = torch.sum(c*u_hat, dim=2, keepdim=True)
            # v = s
            # s.shape = (batch_size, n_out_capsules, 1, n_out_features, 1)
            
            # v = v + self.biases.detach()
            # v = s
            # s.shape = (batch_size, n_out_capsules, 1, n_out_features, 1)
            
            v = CapsuleLayer.squash(v, 3)
            # v.shape = (batch_size, n_out_capsules, 1, n_out_features, 1)
            
            b += torch.matmul(torch.transpose(u_hat, dim0=-2, dim1=-1), v)
            # b.shape = (batch_size, n_out_capsules, n_in_capsules, 1, 1)
            
            c = torch.softmax(b, dim=1)
            # c.shape = (batch_size, n_out_capsules, n_in_capsules, 1, 1)
        
        x = torch.sum(c*x, dim=2, keepdim=True)
        # x = s
        # s.shape = (batch_size, n_out_capsules, 1, n_out_features, 1)
        
        # x = x + self.biases
        # x = s
        # s.shape = (batch_size, n_out_capsules, 1, n_out_features, 1)
        
        x = CapsuleLayer.squash(x, 3)
        # x = v
        # v.shape = (batch_size, n_out_capsules, 1, n_out_features, 1)
        
        # return shape: (batch_size, n_out_capsules, n_out_features)
        return x.view(-1, self.n_out_capsules, self.n_out_features)

class ConvToCapLayer(nn.Module):
    """Converts the convolution layer output into an input for a capsule layer, as mentioned in the original paper.
    Supported input shapes are:
        (batch size or DNE, c, h, w)
    and the output shape:
        (batch size, n_in_capsules_sharing_per_unique, n_in_capsules_unique, n_in_features, 1)
    
    see the doc of CapsuleLayer for the explanation of the parameters.
    """
    def __init__(self, n_in_capsules_unique, n_in_capsules_sharing_per_unique, n_in_features, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        self.n_in_capsules_unique = n_in_capsules_unique
        self.n_in_capsules_sharing_per_unique = n_in_capsules_sharing_per_unique
        self.n_in_capsules = n_in_capsules_unique * n_in_capsules_sharing_per_unique
        self.n_in_features = n_in_features
        
        self.to(device)
    
    def forward(self, x):
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        # input shape: (n, c, h, w) or (c, h, w)

        x = torch.transpose(x, dim0=1, dim1=3)
        # new transposed shape: (n, w, h, c)
        # this needs to be done via transpose, because we want to change the memory layout.
        # otherwise the next line of code won't provide us with the weight sharing we want, or with the capsules we want.
        
        x = x.reshape(-1, self.n_in_capsules_unique, self.n_in_capsules_sharing_per_unique, self.n_in_features, 1)
        # x shape: (-1, n_in_capsules_unique, n_in_capsules_sharing_per_unique, n_in_features, 1)
        # we cannot use view because x isn't contiguous due to the transpose operation before.
        
        x = CapsuleLayer.squash(x, dim=-2)
        # x shape: (-1, n_in_capsules_sharing_per_unique, n_in_capsules_unique, n_in_features, 1)
        
        return x

class CapsuleNetworkCostLayer(nn.Module):
    """Calculates the loss as given in the original paper. Supported input shapes are:
        for input_data          : (batch size, c, h, w)
        for labels              : (batch size, n_classes)
        for capsule_predictions : (batch size, n_out_capsules, n_out_features)
    outputs a single element tensor.
    """
    def __init__(self, alpha=0.0005, lamb=0.5, m_minus=0.1, m_plus=0.9, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        self.alpha = alpha
        self.lamb = lamb
        self.m_minus = m_minus
        self.m_plus = m_plus
        
        self.to(self.device)

    def forward(self, input_data, labels, reconstructions, capsule_predictions):
        margin_cost = self.margin_cost(capsule_predictions, labels)
        reconstruction_cost = self.reconstruction_cost(input_data, reconstructions)
        cost = margin_cost + self.alpha * reconstruction_cost
        return cost
        
    def margin_cost(self, capsule_predictions, labels):
        # capsule_predictions shape: (m, n_out_capsules, n_out_features)
        # labels shape: (m, n_classes)
        # where n_out_capsules = n_classes
        
        norms = torch.linalg.norm(capsule_predictions, dim=2)
        # norms shape: (m, n_classes)
        
        positive_label_losses = labels * torch.relu(self.m_plus - norms)**2
        # positive_label_losses shape: (m, n_classes)
        
        negative_label_losses = (1-labels) * torch.relu(norms - self.m_minus)**2
        # negative_label_losses shape: (m, n_classes)

        losses = torch.sum(positive_label_losses + self.lamb*negative_label_losses, dim=1) 
        # losses shape: (m,)
        
        cost = torch.sum(losses) / losses.shape[0]
        # cost shape: (1,)
        
        return cost
    
    def reconstruction_cost(self, input_data, reconstruction):
        # input_data shape: (n or DNE, c, h, w)
        # reconstruction shape: (n, c, h, w)
        
        input_data = input_data.view(-1, input_data.shape[-3], input_data.shape[-2], input_data.shape[-1])
        # input_data shape: (n, c, h, w)
        
        losses = torch.flatten(reconstruction - input_data, start_dim=1)
        # losses shape: (n, -1)
        
        losses = torch.square(losses)
        # losses shape: (n, -1)
        
        losses = torch.sum(losses, dim=1) / losses.shape[1]
        # losses shape: (n,)
        
        cost = torch.sum(losses) / losses.shape[0]
        # cost shape: (1,)
        
        return cost

class MaskLayer(nn.Module):
    """ Handles the masking mention in the original paper. Supported input shapes are:
        for x: (n, n_classes, n_features)
        for y: (n or DNE, n_classes)
    and the output shape:
        if flatten==False: (n, n_classes, n_features)
        if flatten==True: (n, n_classes * n_features)

    """
    def __init__(self, n_classes, threshold, flatten=True, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.flatten = flatten
        
        self.n_classes = n_classes
        self.threshold = threshold
        
        self.to(device)
    
    def forward(self, x, y):
        # x:capsule layer output, y:label
        # x shape = (n, n_classes, n_features)
        # y.shape = (n or DNE, n_classes) or None
        
        if (self.training == True) and (y == None):
            raise ValueError("In training mode, you need to pass the labels as well for masking.")
        elif (self.training == False) and (y != None):
            raise ValueError("In eval mode, you shouldn't be passing the labels.")
        
        if self.training == True:
            y = y.view(-1, self.n_classes, 1)
            # y.shape: (n, n_classes, 1)
            
        elif self.training == False:
            y = self.extract_onehot_predictions(x.detach())
            # y.shape: (n, n_classes, 1)
        
        x = x * y
        # x.shape: (n, n_classes, n_features)
        
        if self.flatten: x = torch.flatten(x, 1)
        # x.shape: (n, n_classes * n_features)
        
        return x
    
    def extract_onehot_predictions(self, x):
        # x.shape = (n, n_classes, n_features)
        
        y = torch.linalg.norm(x, dim=2)
        # y.shape = (n, n_classes)
        
        y[y<self.threshold] = 0
        y[y!=0] = 1
        # y.shape = (n, n_classes)
        
        # return shape: (n, n_classes, 1)
        return y.view(*y.shape, 1)
