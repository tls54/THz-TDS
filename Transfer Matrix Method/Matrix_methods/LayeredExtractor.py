import torch
import torch.nn as nn
import torch.optim as optim
from .Simulate import simulate_parallel
from time import perf_counter
import numpy as np

# Define loss function for model
def gen_loss_function(y_simulated, y_exp, alpha: float):
    """
    Computes the loss function.
    """
    return alpha * torch.sqrt(torch.nn.functional.mse_loss(y_simulated, y_exp))

## General optimizer to find n_j, k_j, and D_j (thickness) in the time domain
class LayeredExtractor(nn.Module):
    def __init__(self, reference_pulse, experimental_pulse, deltat, layers_init, lr=1e-3):
        super().__init__()
        
        self.reference_pulse = reference_pulse.clone().detach()
        self.experimental_pulse = experimental_pulse.clone().detach()
        self.deltat = deltat
        
        # Extract n, k, and D correctly from layers_init
        self.n_params = nn.ParameterList([nn.Parameter(torch.tensor(np.real(layer[0]), dtype=torch.float32)) for layer in layers_init])
        self.k_params = nn.ParameterList([nn.Parameter(torch.tensor(np.imag(layer[0]), dtype=torch.float32)) for layer in layers_init])
        self.log_D_params = nn.ParameterList([nn.Parameter(torch.tensor(layer[1]).log()) for layer in layers_init])

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Loss history tracking
        self.loss_history = []
        self.best_loss = float('inf')
        
        # Store best parameters
        self.best_n_params = [n.clone().detach() for n in self.n_params]
        self.best_k_params = [k.clone().detach() for k in self.k_params]
        self.best_log_D_params = [log_D.clone().detach() for log_D in self.log_D_params]

    def forward(self):
        """
        Simulates the time-domain response based on current n, k, and D for multiple layers.
        """
        layers = [(n + 1j * k, log_D.exp()) for n, k, log_D in zip(self.n_params, self.k_params, self.log_D_params)]
        y_simulated = simulate_parallel(self.reference_pulse, layers, self.deltat, noise_level=0)[1]
        y_simulated = y_simulated[:len(self.reference_pulse)]
        return y_simulated

    def loss_function(self, y_simulated, alpha):
        """
        Computes the loss function.
        """
        return gen_loss_function(y_simulated, self.experimental_pulse, alpha)

    def optimize(self, num_iterations=100, verbose=True, updates=10, alpha=1):
        print(f'Optimizing for {num_iterations} iterations with loss multiplier {alpha}.')
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            start = perf_counter()
            y_simulated = self.forward()
            self.forward_time = perf_counter() - start

            loss = self.loss_function(y_simulated, alpha=alpha)
            start = perf_counter()
            loss.backward()
            self.backwards_time = perf_counter() - start

            start = perf_counter()
            self.optimizer.step()
            self.optimizer_time = perf_counter() - start
            
            self.loss_history.append(loss.item())
            
            # Store best parameters if loss improves
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_n_params = [n.clone().detach() for n in self.n_params]
                self.best_k_params = [k.clone().detach() for k in self.k_params]
                self.best_log_D_params = [log_D.clone().detach() for log_D in self.log_D_params]

            if verbose and iteration % updates == 0:
                layer_info = ", ".join([f"Layer {j}: n={n.item()}, k={k.item()}, D={log_D.exp().item()}"
                for j, (n, k, log_D) in enumerate(zip(self.n_params, self.k_params, self.log_D_params))])
                print(f"Iteration {iteration}, Loss: {loss.item()}, {layer_info}")

        best_D_params = [log_D.exp().item() for log_D in self.best_log_D_params]
        return [n.item() for n in self.best_n_params], [k.item() for k in self.best_k_params], best_D_params
    




## General optimizer to find n_j and k_j (thickness D is fixed) in the time domain
class LayeredExtractorNK(nn.Module):
    def __init__(self, reference_pulse, experimental_pulse, deltat, layers_init, lr=1e-3):
        super().__init__()
        
        self.reference_pulse = reference_pulse.clone().detach()
        self.experimental_pulse = experimental_pulse.clone().detach()
        self.deltat = deltat
        
        # Extract n, k, and D correctly from layers_init
        self.n_params = nn.ParameterList([nn.Parameter(torch.tensor(np.real(layer[0]), dtype=torch.float32)) for layer in layers_init])
        self.k_params = nn.ParameterList([nn.Parameter(torch.tensor(np.imag(layer[0]), dtype=torch.float32)) for layer in layers_init])
        self.D_values = [layer[1] for layer in layers_init]  # D is now fixed and not a trainable parameter

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Loss history tracking
        self.loss_history = []
        self.best_loss = float('inf')
        
        # Store best parameters
        self.best_n_params = [n.clone().detach() for n in self.n_params]
        self.best_k_params = [k.clone().detach() for k in self.k_params]

    def forward(self):
        """
        Simulates the time-domain response based on current n, k for multiple layers with fixed D.
        """
        layers = [(n + 1j * k, D) for n, k, D in zip(self.n_params, self.k_params, self.D_values)]
        y_simulated = simulate_parallel(self.reference_pulse, layers, self.deltat, noise_level=0.002)[1]
        y_simulated = y_simulated[:len(self.reference_pulse)]
        return y_simulated

    def loss_function(self, y_simulated, alpha):
        """
        Computes the loss function.
        """
        return gen_loss_function(y_simulated, self.experimental_pulse, alpha)

    def optimize(self, num_iterations=100, verbose=True, updates=10, alpha=1):
        print(f'Optimizing for {num_iterations} iterations with loss multiplier {alpha}.')
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            start = perf_counter()
            y_simulated = self.forward()
            self.forward_time = perf_counter() - start

            loss = self.loss_function(y_simulated, alpha=alpha)
            start = perf_counter()
            loss.backward()
            self.backwards_time = perf_counter() - start

            start = perf_counter()
            self.optimizer.step()
            self.optimizer_time = perf_counter() - start
            
            self.loss_history.append(loss.item())
            
            # Store best parameters if loss improves
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_n_params = [n.clone().detach() for n in self.n_params]
                self.best_k_params = [k.clone().detach() for k in self.k_params]

            if verbose and iteration % updates == 0:
                layer_info = ", ".join([f"Layer {j}: n={n.item()}, k={k.item()}, D={D}"
                for j, (n, k, D) in enumerate(zip(self.n_params, self.k_params, self.D_values))])
                print(f"Iteration {iteration}, Loss: {loss.item()}, {layer_info}")

        return [n.item() for n in self.best_n_params], [k.item() for k in self.best_k_params]



## General optimizer to find D_j (thickness) while keeping n_j and k_j fixed in the time domain
class LayeredExtractorD(nn.Module):
    def __init__(self, reference_pulse, experimental_pulse, deltat, layers_init, lr=1e-3):
        super().__init__()
        
        self.reference_pulse = reference_pulse.clone().detach()
        self.experimental_pulse = experimental_pulse.clone().detach()
        self.deltat = deltat
        
        # Extract n, k, and D correctly from layers_init
        self.n_values = [np.real(layer[0]) for layer in layers_init]  # n is fixed
        self.k_values = [np.imag(layer[0]) for layer in layers_init]  # k is fixed
        self.log_D_params = nn.ParameterList([nn.Parameter(torch.tensor(layer[1]).log()) for layer in layers_init])

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Loss history tracking
        self.loss_history = []
        self.best_loss = float('inf')
        
        # Store best parameters
        self.best_log_D_params = [log_D.clone().detach() for log_D in self.log_D_params]

    def forward(self):
        """
        Simulates the time-domain response based on current D for multiple layers with fixed n, k.
        """
        layers = [(n + 1j * k, log_D.exp()) for n, k, log_D in zip(self.n_values, self.k_values, self.log_D_params)]
        y_simulated = simulate_parallel(self.reference_pulse, layers, self.deltat, noise_level=0.002)[1]
        y_simulated = y_simulated[:len(self.reference_pulse)]
        return y_simulated

    def loss_function(self, y_simulated, alpha):
        """
        Computes the loss function.
        """
        return gen_loss_function(y_simulated, self.experimental_pulse, alpha)

    def optimize(self, num_iterations=100, verbose=True, updates=10, alpha=1):
        print(f'Optimizing for {num_iterations} iterations with loss multiplier {alpha}.')
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            start = perf_counter()
            y_simulated = self.forward()
            self.forward_time = perf_counter() - start

            loss = self.loss_function(y_simulated, alpha=alpha)
            start = perf_counter()
            loss.backward()
            self.backwards_time = perf_counter() - start

            start = perf_counter()
            self.optimizer.step()
            self.optimizer_time = perf_counter() - start
            
            self.loss_history.append(loss.item())
            
            # Store best parameters if loss improves
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_log_D_params = [log_D.clone().detach() for log_D in self.log_D_params]

            if verbose and iteration % updates == 0:
                layer_info = ", ".join([f"Layer {j}: n={n}, k={k}, D={log_D.exp().item()}"
                for j, (n, k, log_D) in enumerate(zip(self.n_values, self.k_values, self.log_D_params))])
                print(f"Iteration {iteration}, Loss: {loss.item()}, {layer_info}")

        best_D_params = [log_D.exp().item() for log_D in self.best_log_D_params]
        return best_D_params
