import torch

class FrequencyDependentModel2(torch.nn.Module):
    def __init__(self, w_tensor, d, ICs_n, ICs_k):
        super(FrequencyDependentModel2, self).__init__()
        self.w_tensor = w_tensor
        self.d = d
        self.n = torch.nn.Parameter(torch.full_like(w_tensor, ICs_n, dtype=torch.float32))
        self.k = torch.nn.Parameter(torch.full_like(w_tensor, ICs_k, dtype=torch.float32))
        self.loss_history = []

    def forward(self, Physical_model):
        n_complex = self.n + 1j * self.k
        return Physical_model(n_complex, self.w_tensor, self.d)

    def train_model(self, H_exp, Physical_model, loss_fn, epochs, optimizer):
        loss_value = None  # Initialize loss variable

        def closure():
            optimizer.zero_grad()
            H_pred = self.forward(Physical_model)

            # Avoid log(0) by adding a small epsilon
            eps = 1e-9
            loss = loss_fn((H_pred + eps), (H_exp + eps))

            loss.backward()
            return loss

        for epoch in range(epochs):
            loss_value = optimizer.step(closure).item()  # Store latest loss

            if epoch % 10 == 0:
                self.loss_history.append(loss_value)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss_value}")