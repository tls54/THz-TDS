### Utility files for back-propagation code

## Loss functions

import torch
import numpy as np

c = 299792458   # Speed of light in m/s

## Transfer function
def H_th_function(n_complex, w, length):
    '''
    Inputs
    ------
    n: refractive index
    w: frequency of light being propagated
    length: length of the sample we are modelling
    
    outputs
    -------
    returns: output for the transfer function

    Method
    ------
    Equation for transfer function derived from [Input source here]

    '''

    
    return (4 * n_complex) / ((n_complex + 1) ** 2) * torch.exp(-1j * (n_complex - 1) * w * length/ c)


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



# Function to round to significant figures
def round_to_sig_figs(value, sig_figs):
    if value == 0:
        return 0
    return round(value, sig_figs - int(f"{value:.1e}".split('e')[1]) - 1)


# Function to unwrap tensor angles and return as a tensor
def torch_unwrap(phase:torch.tensor) -> torch.tensor:
    phi = np.unwrap(phase)
    phi = torch.tensor(phi, dtype=torch.float32)
    return phi
