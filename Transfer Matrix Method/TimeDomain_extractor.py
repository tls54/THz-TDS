import torch
import torch.nn as nn
import torch.optim as optim
from Matrix_methods.Simulate import simulate_parallel, simulate_reference
from time import perf_counter

# Define loss function for model
def gen_loss_function(y_simulated, y_exp, alpha:float):
        """
        Computes the loss function.
        """
        return alpha * torch.nn.functional.mse_loss(y_simulated, y_exp)



class TimeDomainExtractor(nn.Module):
    def __init__(self, reference_pulse, experimental_pulse, deltat, n_init, k_init, D_init, lr=1e-3):
        super().__init__()
        
        self.reference_pulse = reference_pulse.clone().detach() 
        self.experimental_pulse = experimental_pulse.clone().detach() 

        self.deltat = deltat

        # Trainable parameters
        self.n = nn.Parameter(torch.tensor(n_init, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(k_init, dtype=torch.float32))
        self.log_D = nn.Parameter(torch.tensor(D_init).log())  # Train log(D) to ensure positivity

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Loss history tracking
        self.loss_history = []
        self.best_loss = float('inf')

        # Store best parameters
        self.best_n = self.n.clone().detach()
        self.best_k = self.k.clone().detach()
        self.log_D = nn.Parameter(torch.tensor(D_init).log())  # Train log(D) to ensure positivity

    # forward pass: uses Matrix transfer methods to generate time domain pulse for predicted params
    def forward(self):
        """
        Simulates the time-domain response based on current n, k, and D.
        """
        D = self.log_D.exp()  # Convert back to ensure positivity
        y_simulated = simulate_parallel(self.reference_pulse, [(self.n + 1j*self.k, D)], self.deltat, noise_level=0.002)[1]

        y_simulated = y_simulated[:len(self.reference_pulse)]
        
        return y_simulated

    def loss_function(self, y_simulated, alpha):
        """
        Computes the loss function.
        """
        return gen_loss_function(y_simulated, self.experimental_pulse, alpha)

    def optimize(self, num_iterations=100, verbose=True, updates=10, alpha=10):
        print(f'Optimizing for {num_iterations} with loss multiplier {alpha}.')
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            stop, start = 0,0

            start = perf_counter()
            y_simulated = self.forward()
            stop = perf_counter()
            self.forward_time = stop - start
            stop, start = 0,0
            
            loss = self.loss_function(y_simulated, alpha=alpha)
            start = perf_counter()
            loss.backward()
            stop = perf_counter()

            self.backwards_time = stop - start
            stop,start = 0,0

            start = perf_counter()
            self.optimizer.step()
            stop = perf_counter()
            self.optimizer_time = stop - start

            self.loss_history.append(loss.item())

            # Store best parameters if loss improves
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_n = self.n.clone().detach()
                self.best_k = self.k.clone().detach()
                self.best_log_D = self.log_D.clone().detach()


            if verbose and iteration % updates == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}, n: {self.n.item()}, k: {self.k.item()}, D: {self.log_D.exp().item()}")

        return self.best_n.item(), self.best_k.item(), self.best_log_D.exp().item()




class TimeDomainExtractorNK(nn.Module):
    def __init__(self, reference_pulse, experimental_pulse, deltat, n_init, k_init, D, lr=1e-3):
        super().__init__()
        
        self.reference_pulse = reference_pulse.clone().detach() 
        self.experimental_pulse = experimental_pulse.clone().detach() 

        self.deltat = deltat

        # Trainable parameters
        self.n = nn.Parameter(torch.tensor(n_init, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(k_init, dtype=torch.float32))
        self.D = D  # Fixed D 

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Loss history tracking
        self.loss_history = []
        self.best_loss = float('inf')

        # Store best parameters
        self.best_n = self.n.clone().detach()
        self.best_k = self.k.clone().detach()


    def forward(self):
        """
        Simulates the time-domain response based on current n, k, and D.
        """
        D = self.D  
        y_simulated = simulate_parallel(self.reference_pulse, [(self.n + 1j*self.k, D)], self.deltat, noise_level=0.002)[1]

        y_simulated = y_simulated[:len(self.reference_pulse)]
        
        return y_simulated

    def loss_function(self, y_simulated):
        """
        Computes the loss function.
        """
        return 10 * torch.nn.functional.mse_loss(y_simulated, self.experimental_pulse)

    def optimize(self, num_iterations=100, verbose=True, updates=10):
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            stop, start = 0,0

            start = perf_counter()
            y_simulated = self.forward()
            stop = perf_counter()
            self.forward_time = stop - start
            stop, start = 0,0
            
            loss = self.loss_function(y_simulated)
            start = perf_counter()
            loss.backward()
            stop = perf_counter()

            self.backwards_time = stop - start
            stop,start = 0,0

            start = perf_counter()
            self.optimizer.step()
            stop = perf_counter()
            self.optimizer_time = stop - start

            self.loss_history.append(loss.item())

            # Store best parameters if loss improves
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_n = self.n.clone().detach()
                self.best_k = self.k.clone().detach()


            if verbose and iteration % updates == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}, n: {self.n.item()}, k: {self.k.item()}")

        return self.best_n.item(), self.best_k.item()
    



## Extractor for known n,k to find thickness

class TimeDomainExtractorD(nn.Module):
    def __init__(self, reference_pulse, experimental_pulse, deltat, n, k, D_init, lr=1e-3):
        super().__init__()
        
        self.reference_pulse = reference_pulse.clone().detach() 
        self.experimental_pulse = experimental_pulse.clone().detach() 

        self.deltat = deltat

        # Trainable parameters
        self.n = n
        self.k = k
        self.log_D = nn.Parameter(torch.tensor(D_init).log())  # Train log(D) to ensure positivity

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Loss history tracking
        self.loss_history = []
        self.best_loss = float('inf')

        # Store best parameters
        self.best_log_D = self.log_D.clone().detach()

    def forward(self):
        """
        Simulates the time-domain response based on current n, k, and D.
        """
        D = self.log_D.exp()  # Convert back to ensure positivity
        y_simulated = simulate_parallel(self.reference_pulse, [(self.n + 1j*self.k, D)], self.deltat, noise_level=0.002)[1]

        y_simulated = y_simulated[:len(self.reference_pulse)]
        
        return y_simulated

    def loss_function(self, y_simulated):
        """
        Computes the loss function.
        """
        return 10 * torch.nn.functional.mse_loss(y_simulated, self.experimental_pulse)

    def optimize(self, num_iterations=1000, verbose=True, updates=100):
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            stop, start = 0,0

            start = perf_counter()
            y_simulated = self.forward()
            stop = perf_counter()
            self.forward_time = stop - start
            stop, start = 0,0
            
            loss = self.loss_function(y_simulated)
            start = perf_counter()
            loss.backward()
            stop = perf_counter()

            self.backwards_time = stop - start
            stop,start = 0,0

            start = perf_counter()
            self.optimizer.step()
            stop = perf_counter()
            self.optimizer_time = stop - start

            self.loss_history.append(loss.item())

            # Store best parameters if loss improves
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_log_D = self.log_D.clone().detach()

            if verbose and iteration % updates == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}, D: {self.log_D.exp().item()}")

        return self.best_log_D.exp().item()