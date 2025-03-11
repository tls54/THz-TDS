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

