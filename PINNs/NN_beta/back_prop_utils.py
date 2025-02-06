### Utility files for back-propagation code

## Loss functions

import torch
import numpy as np

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
    return abs_tf_loss(H_exp, H) + phase_tf_loss(phase_exp, phase)


## Grid search function

def grid_search(n0, k0, d0, H_values, phi_values, freqs_ang, H_th_function, loss, verbose=False):
    """
    Performs a grid search to optimize n, k, and d parameters by minimizing the loss function.

    Parameters:
    - n0 (float): Initial guess for n.
    - k0 (float): Initial guess for k.
    - d0 (float): Initial guess for d.
    - H_values (list/array): Measured amplitude values.
    - phi_values (list/array): Measured phase values.
    - freqs_ang (list/array): Frequency values (angular).
    - H_th_function (function): Function to compute theoretical transfer function.
    - loss (function): Loss function to compare predicted and actual values.

    Returns:
    - best_params (dict): Dictionary containing optimal values for n, k, and d.
    - min_loss (float): Minimum loss achieved.
    """

    min_loss = np.inf
    best_params = {'n': 0, 'k': 0, 'd': 0}

    for ii in range(3):
        n_pred = n0 + ii * 0.01
        for ij in range(3):
            k_pred = k0 + ij * 0.005
            for ik in range(3):
                d_pred = d0 + ik * 0.0001

                # Compute the theoretical transfer function
                tf_values_pred = [H_th_function((n_pred + k_pred * 1j), f, d_pred) for f in freqs_ang]
                H_values_pred = np.abs(tf_values_pred)
                phi_values_pred = np.unwrap(np.angle(tf_values_pred))

                # Compute loss
                l = loss(H_values, H_values_pred, phi_values, phi_values_pred)

                if l < min_loss:
                    min_loss = l
                    best_params = {'n': n_pred, 'k': k_pred, 'd': d_pred}
                
                if verbose:
                    print(f"{n_pred=:.2f}, {k_pred=:.3f}, {d_pred=:.4f}, Loss: {l:.6f}")

    return best_params, min_loss


