### Loss functions for physics informed machine learning 

import numpy as np
import torch


# Define amplitude loss
def abs_tf_loss(H_exp, H):
    """
    Computes the mean absolute error between the absolute values of the 
    experimental and predicted transfer functions, supporting both PyTorch 
    tensors and NumPy arrays.

    Parameters:
    H_exp (torch.Tensor | np.ndarray): Experimental transfer function.
    H (torch.Tensor | np.ndarray): Predicted transfer function.

    Returns:
    torch.Tensor | float: Mean absolute error as a scalar tensor (if inputs are tensors)
                          or a float (if inputs are NumPy arrays).
    """
    if isinstance(H_exp, np.ndarray) and isinstance(H, np.ndarray):
        return np.mean(np.abs(H_exp - H))
    elif isinstance(H_exp, torch.Tensor) and isinstance(H, torch.Tensor):
        return torch.mean(torch.abs(H_exp - H))
    else:
        raise TypeError("Inputs must be both PyTorch tensors or both NumPy arrays.")



# Define phase loss
def phase_tf_loss(phase_exp, phase):
    """
    Computes the mean absolute error between the experimental and predicted phases,
    supporting both PyTorch tensors and NumPy arrays.

    Parameters:
    phase_exp (torch.Tensor | np.ndarray): Experimental phase values.
    phase (torch.Tensor | np.ndarray): Predicted phase values.

    Returns:
    torch.Tensor | float: Mean absolute error as a scalar tensor (if inputs are tensors)
                          or a float (if inputs are NumPy arrays).
    """
    if isinstance(phase_exp, np.ndarray) and isinstance(phase, np.ndarray):
        return np.mean(np.abs(phase_exp - phase))
    elif isinstance(phase_exp, torch.Tensor) and isinstance(phase, torch.Tensor):
        return torch.mean(torch.abs(phase_exp - phase))
    else:
        raise TypeError("Inputs must be both PyTorch tensors or both NumPy arrays.")



# Define total loss
def loss(H_exp, H, phase_exp, phase):
    """
    Computes the total loss as the sum of absolute transfer function loss and 
    phase loss, supporting both PyTorch tensors and NumPy arrays.

    Parameters:
    H_exp (torch.Tensor | np.ndarray): Experimental transfer function.
    H (torch.Tensor | np.ndarray): Predicted transfer function.
    phase_exp (torch.Tensor | np.ndarray): Experimental phase values.
    phase (torch.Tensor | np.ndarray): Predicted phase values.

    Returns:
    torch.Tensor | float: Total loss as a scalar tensor (if inputs are tensors)
                          or a float (if inputs are NumPy arrays).
    """
    return (abs_tf_loss(H_exp, H)) + (phase_tf_loss(phase_exp, phase))



## Define direct loss comparisons
def complex_transfer_loss(H_exp, H_theory, alpha=1.0, beta=1.0):
    """
    Loss function directly comparing complex transfer functions without phase unwrapping.

    Args:
        H_exp (torch.Tensor): Experimental transfer function (complex).
        H_theory (torch.Tensor): Theoretical transfer function (complex).
        alpha (float): Weight for magnitude loss.
        beta (float): Weight for phase difference loss.

    Returns:
        torch.Tensor: Computed loss value.
    """
    # Magnitude loss (Mean Squared Error on amplitude)
    mag_loss = torch.nn.functional.mse_loss(torch.abs(H_exp), torch.abs(H_theory))

    # Phase loss (difference of angles)
    phase_exp = torch.angle(H_exp)
    phase_theory = torch.angle(H_theory)

    # Direct phase difference, avoiding unwrapping
    phase_diff = torch.sin(phase_theory - phase_exp)  # Keeps values between -1 and 1
    phase_loss = torch.mean(phase_diff**2)  # MSE loss on phase difference

    # Weighted sum of losses
    return alpha * mag_loss + beta * phase_loss



def complex_real_imag_loss(H_pred, H_exp):
    """
    Computes loss using real and imaginary parts separately.
    """

    real_loss = torch.nn.functional.mse_loss(H_pred.real, H_exp.real)
    imag_loss = torch.nn.functional.mse_loss(H_pred.imag, H_exp.imag)

    return real_loss + imag_loss



def complex_real_imag_loss_smoothed(H_exp, H_pred, n, k, smoothing_weight):
    """
    Computes loss using real and imaginary parts separately,
    with an added penalty for discontinuities in `n` and `k`.
    """
    real_loss = torch.nn.functional.mse_loss(H_pred.real, H_exp.real)
    imag_loss = torch.nn.functional.mse_loss(H_pred.imag, H_exp.imag)

    # Convert ParameterList to tensor
    n_tensor = torch.stack([param for param in n])
    k_tensor = torch.stack([param for param in k])

    smoothness_penalty = 0
    if len(n_tensor) > 1:  # Apply only if we have multiple frequency points
        smoothness_penalty = torch.sum((n_tensor[1:] - n_tensor[:-1])**2) + \
                             torch.sum((k_tensor[1:] - k_tensor[:-1])**2)

    return real_loss + imag_loss + smoothing_weight * smoothness_penalty



def log_complex_loss(H_exp, H_pred):
    """Computes loss in the log domain to avoid phase wrapping issues."""
    log_H_exp = torch.log(H_exp)  # Compute log of experimental transfer function
    log_H_pred = torch.log(H_pred)  # Compute log of predicted transfer function

    # Compute MSE loss in the log space
    return torch.nn.functional.mse_loss(log_H_pred.real, log_H_exp.real) + torch.nn.functional.mse_loss(log_H_pred.imag, log_H_exp.imag)

# Define a loss function that punishes discontinuities 
def log_complex_loss_smooth(H_exp, H_pred, n, k, lambda_smooth=1e-3):
    """Computes log-space loss with a smoothing constraint."""
    log_H_exp = torch.log(H_exp)
    log_H_pred = torch.log(H_pred)

    # Compute MSE loss in the log space
    loss_mse = torch.nn.functional.mse_loss(log_H_pred.real, log_H_exp.real) + \
               torch.nn.functional.mse_loss(log_H_pred.imag, log_H_exp.imag)

    # Smoothness penalty (finite differences)
    smoothness_penalty = torch.sum((n[:-2] - 2 * n[1:-1] + n[2:]) ** 2) + \
                         torch.sum((k[:-2] - 2 * k[1:-1] + k[2:]) ** 2)

    # Total loss
    loss = loss_mse + lambda_smooth * smoothness_penalty
    return loss