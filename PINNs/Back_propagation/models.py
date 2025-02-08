### Back-propagation models 
import torch
from back_prop_utils import H_th_function

# Define PyTorch Model
class TransferFunctionModel(torch.nn.Module):
    def __init__(self, w_tensor, d):
        super().__init__()
        self.w_tensor = w_tensor  # Store frequency range
        # Directly initialize parameters within expected range
        self.n = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float32))  
        self.k = torch.nn.Parameter(torch.tensor(-0.05, dtype=torch.float32))  
        #self.d = torch.nn.Parameter(torch.tensor(400e-6, dtype=torch.float32))  
        self.d = d

    def forward(self):
        # Ensure parameters satisfy constraints
        n = self.n # Ensure n > 0
        k = self.k  # Ensure k < 0
        #d = self.d  # Ensure d > 0

        n_complex = n + (1j * k)
        H_th = H_th_function(n_complex=n_complex, w=self.w_tensor, length=self.d)
        return H_th