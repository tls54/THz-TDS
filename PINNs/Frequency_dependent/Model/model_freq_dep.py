import torch
import numpy as np

## TODO: 
# - Add optimal param dict
# - Add Scheduler  

class FrequencyDependentModel(torch.nn.Module):
    def __init__(self, w_tensor, d, ICs_n, ICs_k):
        super().__init__()
        self.w_tensor = w_tensor  # Frequency tensor
        self.d = d  # Thickness of the material
        
        # Initialize n and k as trainable parameters (one per frequency point)
        self.n = torch.nn.Parameter(torch.tensor(ICs_n, dtype=torch.float32).repeat(len(w_tensor)))
        self.k = torch.nn.Parameter(torch.tensor(ICs_k, dtype=torch.float32).repeat(len(w_tensor)))
        
        # Store loss history for plotting
        self.loss_history = []
    
    def forward(self, Physical_model):
        """
        Compute the theoretical transfer function given the current n, k values.
        """
        n_complex = self.n + 1j * self.k
        H_pred = Physical_model(n_complex, self.w_tensor, self.d)
        return H_pred
    
    def train_model(self, H_exp, Physical_model, loss_fn, 
                    optimizer=None, epochs=10000, lr=1e-4, verbose=True, updates=500
                    ):
    
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute theoretical transfer function
            H_pred = self.forward(Physical_model)
            
            # Compute loss
            loss = loss_fn(H_exp, H_pred)
            self.loss_history.append(loss.item())
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % updates == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")
        
        return self.loss_history
