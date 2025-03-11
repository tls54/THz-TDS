import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


## Plot training history
def plot_training_progress(loss_plot, n_actual=None, k_actual=None, thickness=None):
    """
    Plots the training progress in a 1×2 grid, showing:
    - Loss over epochs
    - Log loss over epochs

    Args:
        loss_plot (list): List of loss values per epoch.
        thickness (float, optional): Thickness parameter in meters.

    Returns:
        None
    """

    # Define epochs range
    epochs = range(len(loss_plot))

    # Compute log loss (avoiding log(0) issues)
    log_loss = np.log(np.array(loss_plot) + 1e-8)

    # Find epoch with minimum loss
    min_epoch = np.argmin(loss_plot)

    # Set Seaborn theme
    sns.set_theme(style="darkgrid")

    # Create a 2×2 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Plot raw loss
    sns.lineplot(x=epochs, y=loss_plot, ax=axs[0], label="Loss")
    axs[0].scatter(min_epoch, loss_plot[min_epoch], color="red", 
                      label=f"Min Loss: {min(loss_plot):.2f} @ Epoch {min_epoch}", 
                      edgecolor="black", zorder=3)
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot log loss
    sns.lineplot(x=epochs, y=log_loss, ax=axs[1], color="tab:orange", label="Log Loss")
    axs[1].set_ylabel("Log Loss")
    axs[1].legend()

    # Add overall title
    plt.suptitle("Training Progress: Loss")

    # Conditionally add text annotation if all true values are available
    if n_actual is not None and k_actual is not None and thickness is not None:
        fig.text(0.05, 0.99, f'n_actual={n_actual:.3f}, k_actual={k_actual:.3f}, d={1e6*thickness:.1f}µm', 
                 verticalalignment='top', horizontalalignment='left', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

    # Improve layout
    plt.tight_layout()
    plt.show()


## Define standard method for plotting transfer function data
def plot_transfer(frequencies, absolute_values, phase_values, absolute_values_clean=None, phase_values_clean=None, params=None):
    """
    Plots the absolute value and unwrapped phase of a transfer function.
    
    Args:
        frequencies (array-like): Frequency values in THz.
        absolute_values (array-like): Noisy absolute values of the transfer function.
        phase_values (array-like): Noisy unwrapped phase values of the transfer function (in radians).
        absolute_values_clean (array-like, optional): Clean absolute values for comparison.
        phase_values_clean (array-like, optional): Clean phase values for comparison.
        params (list or None, optional): List containing [n, k, thickness] if available.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Plot phase values
    axs[0].scatter(frequencies, phase_values, s=8, label='Noisy phase values')
    if phase_values_clean is not None:
        axs[0].scatter(frequencies, phase_values_clean, s=8, label='Clean phase values')
    axs[0].set_title('Unwrapped phase of transfer function')
    axs[0].set_xlabel('Frequencies [THz]')
    axs[0].set_ylabel('Angle [Rad]')
    axs[0].legend()

    # Plot absolute values
    axs[1].scatter(frequencies, absolute_values, s=8, label='Noisy absolute values')
    if absolute_values_clean is not None:
        axs[1].scatter(frequencies, absolute_values_clean, s=8, label='Clean absolute values')
    axs[1].set_title('Absolute value of transfer function')
    axs[1].set_xlabel('Frequencies [THz]')
    axs[1].set_ylabel('|H|')
    axs[1].legend()

    # Annotate with n, k, d values if provided
    if params is not None and len(params) == 3:
        n, k, d = params
        fig.text(0.05, 0.99, f'n={n:.3f}, k={k:.3f}, d={1e6*d:.1f}µm', 
                 verticalalignment='top', horizontalalignment='left', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

    plt.show()


## Define plotting method to compare contributions to loss
def plot_loss_contributions(phase_loss, abs_loss):
    """
    Plots the phase and amplitude loss contributions over training iterations.

    Args:
        phase_loss (list or np.array): Phase loss values per iteration.
        abs_loss (list or np.array): Amplitude loss values per iteration.

    Returns:
        None
    """

    # Define iteration range
    iterations = range(len(phase_loss))

    # Find iteration with minimum total loss
    total_loss = np.array(phase_loss) + np.array(abs_loss)
    min_iter = np.argmin(total_loss)

    # Set Seaborn theme
    sns.set_theme(style="darkgrid")

    # Create a 1×2 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Plot phase loss
    sns.lineplot(x=iterations, y=phase_loss, ax=axs[0], label="Phase Loss", color="b")
    axs[0].scatter(min_iter, phase_loss[min_iter], color="red", 
                   label=f"Min: {phase_loss[min_iter]:.4f} @ Iter {min_iter}",
                   edgecolor="black", zorder=3)
    axs[0].set_ylabel("Phase Loss")
    axs[0].set_xlabel("Iterations")
    axs[0].legend()

    # Plot amplitude loss
    sns.lineplot(x=iterations, y=abs_loss, ax=axs[1], label="Amplitude Loss", color="tab:orange")
    axs[1].scatter(min_iter, abs_loss[min_iter], color="red", 
                   label=f"Min: {abs_loss[min_iter]:.4f} @ Iter {min_iter}",
                   edgecolor="black", zorder=3)
    axs[1].set_ylabel("Amplitude Loss")
    axs[1].set_xlabel("Iterations")
    axs[1].legend()

    # Add overall title
    plt.suptitle("Phase and Amplitude Loss Contributions")

    # Improve layout
    plt.tight_layout()
    plt.show()


## Define method for plotting comparison of experimental data and reconstructed transfer function
def plot_comparison(frequencies, exp_abs, exp_phase, pred_abs, pred_phase, params=None):
    """
    Plots the absolute value and unwrapped phase of experimental and predicted transfer functions.
    
    Args:
        frequencies (array-like): Frequency values in THz.
        exp_abs (array-like): Experimental absolute values of the transfer function.
        exp_phase (array-like): Experimental unwrapped phase values (in radians).
        pred_abs (array-like): Predicted absolute values of the transfer function.
        pred_phase (array-like): Predicted unwrapped phase values (in radians).
        params (list or None, optional): List containing [n, k, thickness] if available.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Plot phase values
    axs[0].scatter(frequencies, exp_phase, s=8, label='Experimental phase', alpha=0.7)
    axs[0].scatter(frequencies, pred_phase, s=8, label='Predicted phase', alpha=0.7)
    axs[0].set_title('Unwrapped phase of transfer function')
    axs[0].set_xlabel('Frequencies [THz]')
    axs[0].set_ylabel('Angle [Rad]')
    axs[0].legend()

    # Plot absolute values
    axs[1].scatter(frequencies, exp_abs, s=8, label='Experimental |H|', alpha=0.7)
    axs[1].scatter(frequencies, pred_abs, s=8, label='Predicted |H|', alpha=0.7)
    axs[1].set_title('Absolute value of transfer function')
    axs[1].set_xlabel('Frequencies [THz]')
    axs[1].set_ylabel('|H|')
    axs[1].legend()

    # Annotate with n, k, d values if provided
    if params is not None and len(params) == 3:
        n, k, d = params
        fig.text(0.05, 0.99, f'n={n:.3f}, k={k:.3f}, d={1e6*d:.1f}µm', 
                 verticalalignment='top', horizontalalignment='left', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

    plt.show()


## Plot frequency dependence of material parameters
def plot_material_params(frequencies, n, k):
    
    return