import torch
import numpy as np
from skopt import gp_minimize
from .Simulate import simulate_parallel



# Define loss function for model
def gen_loss_function(y_simulated, y_exp, alpha: float):
    return alpha * torch.sqrt(torch.nn.functional.mse_loss(y_simulated, y_exp))



## Model to find D values given n,k or arbitrary layers
class BayesianLayeredExtractorD():
    def __init__(self, reference_pulse, experimental_pulse, deltat, layers_init, lr=1e-2):
        super().__init__()
        
        self.reference_pulse = reference_pulse.clone().detach()
        self.experimental_pulse = experimental_pulse.clone().detach()
        self.deltat = deltat
        
        self.n_values = [np.real(layer[0]) for layer in layers_init]  # Fixed refractive index
        self.k_values = [np.imag(layer[0]) for layer in layers_init]  # Fixed extinction coefficient
        self.D_init = [layer[1] for layer in layers_init]  # Initial thickness

        
        self.optimizer = None  # Placeholder for later use
    def loss_function(self, y_simulated, alpha):
        return gen_loss_function(y_simulated, self.experimental_pulse, alpha)
    
    def bayesian_optimization(self, n_calls=20, alpha=1):
        print("Starting Bayesian Optimization...")
        def objective(D_values):
            layers = [(n + 1j * k, D) for (n, k), D in zip(zip(self.n_values, self.k_values), D_values)]
            y_simulated = simulate_parallel(self.reference_pulse, layers, self.deltat, noise_level=0)[1][:len(self.experimental_pulse)]
            return gen_loss_function(y_simulated, self.experimental_pulse, alpha).item()
        
        # Define bounds around the initial guess ± 0.15e-3
        bounds = [(D - 0.15e-3, D + 0.15e-3) for D in self.D_init]
        
        result = gp_minimize(objective, bounds, n_calls=n_calls, random_state=42)
        best_D_params = result.x
        
        print("Bayesian Optimization complete.")

        return best_D_params



## Model to find n, k values given a known D
class BayesianLayeredExtractorNK():
    def __init__(self, reference_pulse, experimental_pulse, deltat, layers_init, lr=1e-2):
        super().__init__()
        
        self.reference_pulse = reference_pulse.clone().detach()
        self.experimental_pulse = experimental_pulse.clone().detach()
        self.deltat = deltat
        
        self.D_values = [layer[1] for layer in layers_init]  # Fixed thickness
        self.n_init = [np.real(layer[0]) for layer in layers_init]  # Initial refractive index
        self.k_init = [np.imag(layer[0]) for layer in layers_init]  # Initial extinction coefficient
    
    def loss_function(self, y_simulated, alpha):
        return gen_loss_function(y_simulated, self.experimental_pulse, alpha)
    
    def bayesian_optimization(self, n_calls=20, alpha=1):
        print("Starting Bayesian Optimization for n and k...")
        
        def objective(nk_values):
            n_values = nk_values[:len(nk_values)//2]  # First half is n
            k_values = nk_values[len(nk_values)//2:]  # Second half is k
            
            layers = [(n + 1j * k, D) for n, k, D in zip(n_values, k_values, self.D_values)]
            y_simulated = simulate_parallel(self.reference_pulse, layers, self.deltat, noise_level=0)[1][:len(self.experimental_pulse)]
            return gen_loss_function(y_simulated, self.experimental_pulse, alpha).item()
        
        # Define bounds around the initial guess ± 0.1 for n and ± 0.01 for k
        bounds_n = [(n - 0.1, n + 0.1) for n in self.n_init]
        bounds_k = [(k - 0.01, k + 0.01) for k in self.k_init]
        bounds = bounds_n + bounds_k  # Combine into a single list
        
        result = gp_minimize(objective, bounds, n_calls=n_calls, random_state=42)
        best_nk_params = result.x
        
        print("Bayesian Optimization complete.")
        
        best_n_values = best_nk_params[:len(best_nk_params)//2]
        best_k_values = best_nk_params[len(best_nk_params)//2:]
        
        return [(n + 1j * k, D) for n, k, D in zip(best_n_values, best_k_values, self.D_values)]

