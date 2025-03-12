import torch
import numpy as np
from .back_prop_utils import H_th_function
from .loss_functions import abs_tf_loss, phase_tf_loss, loss
from .plotting_utils import plot_training_progress
from .loss_functions import loss, abs_tf_loss, phase_tf_loss

# TODO: Add doc strings to all modules


class BaseModel(torch.nn.Module):
    def __init__(self, w_tensor, d, ICs_n, ICs_k):
        super().__init__()
        self.w_tensor = w_tensor  # Frequency tensor
        self.d = d  # Fixed thickness
        
        # Extend frequency range for consistency
        # Added conditional to check if frequencies already start at 0
        if w_tensor[0] != 0:
            delta_w = w_tensor[1] - w_tensor[0]  # Assuming uniform spacing
            extended_w = torch.arange(0, w_tensor[-1] + delta_w, delta_w, device=w_tensor.device)
            self.full_w_tensor = extended_w
            
        else:
            self.full_w_tensor = self.w_tensor
        
        # Store best parameters found during training
        self.best_params = {'n': None, 'k': None, 'loss': float('inf')}
        
        # Initialize loss storage
        self.abs_loss_history = []
        self.phase_loss_history = []
    
    def forward(self):
        # Can't call a forward step for the base model.
        raise NotImplementedError("Subclasses must implement the forward method")
    
    def train_model(self, loss_fn, H_values, phi_values, optimizer=None, scheduler=None, epochs=10000, lr=1e-4, verbose=True, updates=500):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.loss_history = []
        self.abs_loss_history = []
        self.phase_loss_history = []
        
        for epoch in range(epochs):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform Forward step
            H_pred_amp, H_pred_phase_unwrapped = self()
            
            # Calculate loss for back propagation
            total_loss = loss_fn(H_values, H_pred_amp, phi_values, H_pred_phase_unwrapped)
            # Calculate losses for components
            amplitude_loss = abs_tf_loss(H_values, H_pred_amp)
            phase_loss = phase_tf_loss(phi_values, H_pred_phase_unwrapped)
            
            # Add to history for plotting
            self.loss_history.append(total_loss.item())
            self.abs_loss_history.append(amplitude_loss.item())
            self.phase_loss_history.append(phase_loss.item())
            
            # Check for best params
            if total_loss.item() < self.best_params['loss']:
                self.best_params['n'] = self.n.detach().cpu().numpy()
                self.best_params['k'] = self.k.detach().cpu().numpy()
                self.best_params['loss'] = total_loss.item()
            
            # Calculate backward steps and move optimizer forwards
            total_loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(total_loss)
                else:
                    scheduler.step()
            
            if (verbose and epoch % updates == 0) or (verbose and epoch == epochs-1):
                current_lr = optimizer.param_groups[0]['lr'] if scheduler is None else scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}: Total Loss = {total_loss.item()}, Learning Rate = {current_lr}")
        
        # TODO: Add conditional to prevent printing full array in freq dependent model
        if verbose:
            print(f"Best parameters found (Lowest Loss: {self.best_params['loss']}):")
            print(f"n: {self.best_params['n']}")
            print(f"k: {self.best_params['k']}")
        
        return self.loss_history, self.best_params 
    
    def plot_training_curves(self):
        plot_training_progress(self.loss_history, None, None)

## Average n,k model
# TODO: add doc string
class AverageTransferFunctionModel(BaseModel):
    def __init__(self, w_tensor, d, ICs_n, ICs_k):
        super().__init__(w_tensor, d, ICs_n, ICs_k)
        self.n = torch.nn.Parameter(torch.tensor(ICs_n, dtype=torch.float32))  
        self.k = torch.nn.Parameter(torch.tensor(ICs_k, dtype=torch.float32))  
    
    def forward(self):
        n_complex = self.n + 1j * self.k
        H_full = H_th_function(n_complex=n_complex, w=self.full_w_tensor, length=self.d)
        
        H_abs = torch.abs(H_full)
        H_phase = torch.angle(H_full)
        H_phase_unwrapped = np.unwrap(H_phase.detach().cpu().numpy())
        H_phase_unwrapped = torch.tensor(H_phase_unwrapped, dtype=torch.float32).to(H_full.device)
        
        H_abs_truncated = H_abs[-len(self.w_tensor):]
        H_phase_unwrapped_truncated = H_phase_unwrapped[-len(self.w_tensor):]
        
        return H_abs_truncated, H_phase_unwrapped_truncated

