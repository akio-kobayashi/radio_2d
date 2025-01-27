import torch
from einops import rearrange

def negentropy(x, G=torch.log(torch.cosh)):
    """
    Compute an approximation of negentropy for a batch of tensors.
    
    Args:
        x (Tensor): Input tensor of shape (batch, feature, time).
        G (callable): Non-linear function for negentropy approximation.
    
    Returns:
        Tensor: Negentropy for each sample in the batch.
    """
    # Flatten along feature and time dimensions
    x = rearrange(x, 'b f t -> b (f t)')
    # Normalize input
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-8
    x_normalized = (x - mean) / std

    # Compute G(x)
    negentropy = G(x_normalized).mean(dim=1)
    return negentropy

class MaximumNegentropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, reference):
        """
        Compute Maximum Negentropy Loss between prediction and reference.
        
        Args:
            prediction (Tensor): Predicted tensor (batch, feature, time).
            reference (Tensor): Reference tensor (batch, feature, time).
        
        Returns:
            Tensor: Negentropy loss value.
        """
        def G(x):
            return torch.log(torch.cosh(x))  # Example of G(x)

        # Compute negentropy
        negentropy_pred = negentropy(prediction, G)
        negentropy_ref = negentropy(reference, G)

        # Loss: Mean squared error of negentropies
        loss = torch.nn.functional.mse_loss(negentropy_pred, negentropy_ref)
        return loss
