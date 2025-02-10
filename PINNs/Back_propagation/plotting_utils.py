import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_training_progress(loss_plot, n_vals, k_vals, n_actual, k_actual, thickness):
    """
    Plots the training progress in a 2×2 grid, showing:
    - Loss over epochs
    - Log loss over epochs
    - Evolution of parameter n
    - Evolution of parameter k

    Args:
        loss_plot (list): List of loss values per epoch.
        n_vals (list): List of n values per epoch.
        k_vals (list): List of k values per epoch.
        n_actual (float): Actual value of n.
        k_actual (float): Actual value of k.
        d (float): Thickness parameter in meters.

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
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    # Plot raw loss
    sns.lineplot(x=epochs, y=loss_plot, ax=axs[0, 0], label="Loss")
    axs[0, 0].scatter(min_epoch, loss_plot[min_epoch], color="red", 
                      label=f"Min Loss: {min(loss_plot):.2f} @ Epoch {min_epoch}", 
                      edgecolor="black", zorder=3)
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # Plot log loss
    sns.lineplot(x=epochs, y=log_loss, color="tab:orange", ax=axs[0, 1], label="Log Loss")
    axs[0, 1].set_ylabel("Log Loss")
    axs[0, 1].legend()

    # Plot n values
    sns.lineplot(x=epochs, y=n_vals, ax=axs[1, 0], label="n values")
    axs[1, 0].scatter(min_epoch, n_vals[min_epoch], color="red", 
                      label=f"Min Loss: {n_vals[min_epoch]:.4f} @ Epoch {min_epoch}", 
                      edgecolor="black", zorder=3)
    axs[1, 0].axhline(y=n_actual, color="red", linestyle="--", label="Actual n")  
    axs[1, 0].set_ylabel("n")
    axs[1, 0].legend()

    # Plot k values
    sns.lineplot(x=epochs, y=k_vals, color="tab:orange", ax=axs[1, 1], label="k values")
    axs[1, 1].scatter(min_epoch, k_vals[min_epoch], color="red", 
                      label=f"Min Loss: {k_vals[min_epoch]:.4f} @ Epoch {min_epoch}", 
                      edgecolor="black", zorder=3)
    axs[1, 1].axhline(y=k_actual, color="red", linestyle="--", label="Actual k")  
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("k")
    axs[1, 1].legend()

    # Add overall title
    plt.suptitle("Training Progress: Loss and Parameter Evolution")

    # Add a label at the top-left of the entire figure (outside the subplots)
    fig.text(0.05, 0.99, f'n_actual={n_actual:.3f}, k_actual={k_actual:.3f}, d={1e6*thickness:.1f}µm', 
             verticalalignment='top', horizontalalignment='left', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

    # Improve layout
    plt.tight_layout()
    plt.show()