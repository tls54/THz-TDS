import torch
import numpy as np
from back_prop_utils import H_th_function
from plotting_utils import plot_training_progress

class TransferFunctionModel(torch.nn.Module):
    def __init__(self, w_tensor, d, ICs:list):
        """
        
        """
        super().__init__()
        self.w_tensor = w_tensor  
        self.n = torch.nn.Parameter(torch.tensor(ICs[0], dtype=torch.float32))  
        self.k = torch.nn.Parameter(torch.tensor(ICs[1], dtype=torch.float32))  
        self.d = d  # Fixed thickness

        # Store best parameters found during training
        self.best_params = {'n': self.n.item(), 'k': self.k.item(), 'loss': float('inf')}

    def forward(self):
        n_complex = self.n + 1j * self.k
        return H_th_function(n_complex=n_complex, w=self.w_tensor, length=self.d)

    def train_model(self, loss_fn, H_values, phi_values, optimizer=None, epochs=10000, lr=1e-4, verbose=True):
        """
        Train the model using backpropagation and track the best parameters.

        Parameters:
        - loss_fn: Function to compute the loss.
        - H_values: Target amplitude values.
        - phi_values: Target phase values.
        - optimizer: PyTorch optimizer (default: Adam with lr=1e-4).
        - epochs: Number of training iterations (default: 10000).
        - lr: Learning rate (default: 1e-4).
        - verbose: If True, prints progress every 500 epochs.

        Returns:
        - loss_plot: List of loss values over epochs.
        - n_vals: List of optimized n values over epochs.
        - k_vals: List of optimized k values over epochs.
        - best_params: Dictionary containing {'n': best_n, 'k': best_k, 'loss': lowest_loss}.
        """

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            
        self.loss_history = []
        self.n_vals = []
        self.k_vals = []

        for epoch in range(epochs):
            self.n_vals.append(self.n.item())
            self.k_vals.append(self.k.item())

            optimizer.zero_grad()

            H_pred = self()
            H_pred_amp = torch.abs(H_pred)
            H_pred_phase = torch.angle(H_pred)

            H_pred_phase_unwrapped = np.unwrap(H_pred_phase.detach().cpu().numpy())
            H_pred_phase_unwrapped = torch.tensor(H_pred_phase_unwrapped, dtype=torch.float32).to(H_pred.device)

            loss_value = loss_fn(H_values, H_pred_amp, phi_values, H_pred_phase_unwrapped)
            self.loss_history.append(loss_value.item())

            # Update best parameters if this is the lowest loss seen
            if loss_value.item() < self.best_params['loss']:
                self.best_params['n'] = self.n.item()
                self.best_params['k'] = self.k.item()
                self.best_params['loss'] = loss_value.item()

            loss_value.backward()
            optimizer.step()

            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss_value.item()}")

        if verbose:
            print(f"Final n: {self.n.item()}, Final k: {self.k.item()}")
            print(f"Best n: {self.best_params['n']}, Best k: {self.best_params['k']} (Lowest Loss: {self.best_params['loss']})")

        return self.loss_history, self.n_vals, self.k_vals, self.best_params

    # Define easy plotting method to quickly call plots
    def plot_training_curves(self, n_actual, k_actual, thickness):
        plot_training_progress(self.loss_history, self.n_vals, self.k_vals, n_actual, k_actual, thickness)