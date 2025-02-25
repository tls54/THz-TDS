import torch
import numpy as np
from back_prop_utils import H_th_function
from plotting_utils import plot_training_progress
from loss_functions import *


class TransferFunctionModel(torch.nn.Module):
    def __init__(self, w_tensor, d, ICs: list):
        """
        Initialize the model with the given parameters.
        """
        super().__init__()
        self.w_tensor = w_tensor  
        self.n = torch.nn.Parameter(torch.tensor(ICs[0], dtype=torch.float32))  
        self.k = torch.nn.Parameter(torch.tensor(ICs[1], dtype=torch.float32))  
        self.d = d  # Fixed thickness

        # Store best parameters found during training
        self.best_params = {'n': self.n.item(), 'k': self.k.item(), 'loss': float('inf')}

        # Initialize loss components storage
        self.abs_loss_history = []  # Amplitude loss history
        self.phase_loss_history = []  # Phase loss history

    def forward(self):
        n_complex = self.n + 1j * self.k
        return H_th_function(n_complex=n_complex, w=self.w_tensor, length=self.d)

    def train_model(self, loss_fn, H_values, phi_values, optimizer=None, scheduler=None, epochs=10000, lr=1e-4, verbose=True):
        """
        Train the model using back-propagation and track the best parameters.
        Supports an optional learning rate scheduler.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Initialize loss lists for analysis
        self.loss_history = []
        self.abs_loss_history = []
        self.phase_loss_history = []
        self.n_vals = []
        self.k_vals = []

        # Main training loop
        for epoch in range(epochs):
            self.n_vals.append(self.n.item())
            self.k_vals.append(self.k.item())
            
            # Zero the gradient
            optimizer.zero_grad()

            # Call forward step (same as self.forward)
            H_pred = self()

            # Split predictions into components
            H_pred_amp = torch.abs(H_pred)
            H_pred_phase = torch.angle(H_pred)

            # Unwrap phase
            H_pred_phase_unwrapped = np.unwrap(H_pred_phase.detach().cpu().numpy())
            H_pred_phase_unwrapped = torch.tensor(H_pred_phase_unwrapped, dtype=torch.float32).to(H_pred.device)

            # Compute total loss, amplitude loss, and phase loss
            total_loss = loss_fn(H_values, H_pred_amp, phi_values, H_pred_phase_unwrapped)
            amplitude_loss = abs_tf_loss(H_values, H_pred_amp)
            phase_loss = phase_tf_loss(phi_values, H_pred_phase_unwrapped)

            # Store losses
            self.loss_history.append(total_loss.item())
            self.abs_loss_history.append(amplitude_loss.item())
            self.phase_loss_history.append(phase_loss.item())

            # Update best parameters if this is the lowest loss seen
            if total_loss.item() < self.best_params['loss']:
                self.best_params['n'] = self.n.item()
                self.best_params['k'] = self.k.item()
                self.best_params['loss'] = total_loss.item()

            # Perform back-propagation
            total_loss.backward()
            optimizer.step()

            # Step the scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(total_loss)  # ReduceLROnPlateau requires a loss value
                else:
                    scheduler.step()  # Other schedulers use step() without arguments

            # Print progress every 500 epochs
            if verbose and epoch % 500 == 0:
                current_lr = optimizer.param_groups[0]['lr'] if scheduler is None else scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}: Total Loss = {total_loss.item()}, Learning Rate = {current_lr}")

        if verbose:
            print(f"Final n: {self.n.item()}, Final k: {self.k.item()}")
            print(f"Best n: {self.best_params['n']}, Best k: {self.best_params['k']} (Lowest Loss: {self.best_params['loss']})")

        # Return loss history and best parameters
        return self.loss_history, self.best_params

    # Define easy plotting method to quickly call plots
    def plot_training_curves(self, n_actual=None, k_actual=None, thickness=None):
        plot_training_progress(self.loss_history, self.n_vals, self.k_vals, n_actual, k_actual, thickness)