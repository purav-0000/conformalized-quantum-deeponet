import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch


# --- Callbacks and Schedulers ---

class LRLogger(dde.callbacks.Callback):
    """
    A DeepXDE callback to log the current learning rate at specified intervals.

    Args:
        display_every (int): The frequency (in epochs) at which to log the LR.
    """
    def __init__(self, display_every: int = 1000):
        super().__init__()
        self.display_every = display_every

    def on_epoch_end(self):
        """Logs the learning rate at the end of an epoch."""
        current_epoch = self.model.train_state.epoch
        if current_epoch > 0 and current_epoch % self.display_every == 0:
            current_lr = self.model.opt.param_groups[0]['lr']
            logging.info(f"└─> [LR at Epoch {current_epoch}]: {current_lr:.2e}")


def create_decay_and_hold_scheduler(initial_lr: float, gamma: float, min_lr: float) -> Callable[[int], float]:
    """
    Creates a lambda scheduler that decays the LR multiplicatively until a minimum is reached.

    Args:
        initial_lr (float): The starting learning rate.
        gamma (float): The multiplicative decay factor (e.g., 0.99).
        min_lr (float): The minimum learning rate to hold after sufficient decay.

    Returns:
        Callable[[int], float]: A scheduler function that takes a step and returns a factor.
    """
    min_factor = min_lr / initial_lr

    def scheduler(step: int) -> float:
        """Calculates the multiplicative factor for the current step."""
        decay_factor = gamma ** step
        return max(decay_factor, min_factor)
    return scheduler


# --- Plotting Utilities ---

def plot_training_inputs(
        x_train: Tuple[np.ndarray, np.ndarray],
        y_train: np.ndarray,
        model_dir: Path,
        x_train_plot: np.ndarray,
        num_samples: int = 3
):
    """
    Visualizes and saves a few random samples of the data being fed to the model.

    Args:
        x_train (Tuple[np.ndarray, np.ndarray]): The tuple of (branch, trunk) inputs.
        y_train (np.ndarray): The ground truth output data.
        model_dir (Path): The directory to save the plots in.
        x_train_plot (np.ndarray): The high-resolution coordinates for the branch input.
        num_samples (int): The number of random samples to plot.
    """
    plot_dir = model_dir / "training_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Generating model input plots in: {plot_dir}")

    indices = np.random.choice(len(x_train[0]), num_samples, replace=False)
    is_online = y_train.shape[-1] == 1

    for i, index in enumerate(indices):
        with plt.style.context('seaborn-v0_8-deep'):
            fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=False)
            fig.suptitle(f"Training Input Sample (Index: {index})", fontsize=14)

            branch_input = x_train[0][index, :-1]  # Exclude the augmented feature
            branch_norm = x_train[0][index, -1]

            if is_online:
                branch_coords = x_train_plot[index]
                dt = branch_coords[1] - branch_coords[0]
                truth_coord = branch_coords[-1] + dt

                axes[0].plot(branch_coords, branch_input, 'o--', color='purple', label='Branch Input (u)')
                axes[0].plot(truth_coord, y_train[index], '*', markersize=12, color='blue', label='Ground Truth (y)')
                axes[0].set_title("Branch Input and Ground Truth")
                axes[0].legend()
                axes[0].grid(True, linestyle='--')
                axes[0].set_ylabel("Value")
                axes[0].text(0.95, 0.01, f'Norm Feature: {branch_norm:.2f}',
                            verticalalignment='bottom', horizontalalignment='right',
                            transform=axes[0].transAxes, color='gray', fontsize=10)
                axes[0].set_xlabel("Time or Coordinate")

                axes[1].remove()


            else:  # Offline
                branch_coords = x_train_plot
                trunk_coords = x_train[1][:, 0]

                # Subplot 1: Branch Input
                axes[0].plot(branch_coords, branch_input, '--', color='purple', label='Branch Input (u)')
                axes[0].set_title("Branch Input")
                axes[0].legend()
                axes[0].grid(True, linestyle='--')
                axes[0].set_ylabel("Value")
                axes[0].text(0.95, 0.01, f'Norm Feature: {branch_norm:.2f}',
                            verticalalignment='bottom', horizontalalignment='right',
                            transform=axes[0].transAxes, color='gray', fontsize=10)

                # Subplot 2: Ground Truth
                axes[1].plot(trunk_coords, y_train[index], '-', color='blue', label='Ground Truth (y)')
                axes[1].set_title("Ground Truth")
                axes[1].legend()
                axes[1].grid(True, linestyle='--')
                axes[1].set_xlabel("Coordinate")
                axes[1].set_ylabel("Value")

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlap
            fig.savefig(plot_dir / f"input_sample_{i}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)


def plot_model_outputs(
        model: dde.Model,
        x_test: Tuple[np.ndarray, np.ndarray],
        y_test: np.ndarray,
        model_dir: Path,
        x_test_plot: np.ndarray,
        num_samples: int = 3
):
    """
    Generates a suite of plots to evaluate model performance on the test set.

    This includes an error distribution histogram and plots of the best, worst,
    and random predictions.

    Args:
        model (dde.Model): The trained DeepXDE model.
        x_test (Tuple[np.ndarray, np.ndarray]): The tuple of (branch, trunk) test inputs.
        y_test (np.ndarray): The ground truth test outputs.
        model_dir (Path): The directory to save the plots in.
        x_test_plot (np.ndarray): The high-resolution coordinates for the branch input.
        num_samples (int): The number of best/worst/random samples to plot.
    """
    plot_dir = model_dir / "training_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Generating model output plots in: {plot_dir}")

    with torch.no_grad():
        y_pred = model.predict(x_test)

    # 1. Calculate errors and plot distribution
    errors = np.linalg.norm(y_pred - y_test, axis=1) / np.linalg.norm(y_test, axis=1)
    _plot_error_distribution(errors, plot_dir)

    # 2. Find and plot best/worst predictions
    sorted_indices = np.argsort(errors)
    worst_indices = sorted_indices[-num_samples:]
    best_indices = sorted_indices[:num_samples]

    _plot_predictions_subplot(
        indices=worst_indices, title_prefix="Worst", color='red',
        x_test=x_test, y_test=y_test, y_pred=y_pred, errors=errors,
        x_test_plot=x_test_plot, save_path=plot_dir / "worst_predictions.png"
    )
    _plot_predictions_subplot(
        indices=best_indices, title_prefix="Best", color='green',
        x_test=x_test, y_test=y_test, y_pred=y_pred, errors=errors,
        x_test_plot=x_test_plot, save_path=plot_dir / "best_predictions.png"
    )


def _plot_error_distribution(errors: np.ndarray, plot_dir: Path):
    """Plots and saves a histogram of the L2 relative errors."""
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=50, alpha=0.8, color='steelblue')
        ax.set_title("Distribution of L2 Relative Errors on Test Set")
        ax.set_xlabel("L2 Relative Error")
        ax.set_ylabel("Number of Samples")
        ax.set_yscale('log')
        ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean Error: {np.mean(errors):.2%}')
        ax.legend()
        fig.savefig(plot_dir / "error_distribution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def _plot_predictions_subplot(
        indices: np.ndarray, title_prefix: str, color: str,
        x_test: Tuple, y_test: np.ndarray, y_pred: np.ndarray, errors: np.ndarray,
        x_test_plot: np.ndarray, save_path: Path
):
    """
    Creates a single figure with multiple subplots for a set of predictions.
    """
    num_plots = len(indices)
    is_online = y_test.shape[-1] == 1

    with plt.style.context('seaborn-v0_8-deep'):
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots),
                                 sharex=True if not is_online else False)
        if num_plots == 1: axes = [axes]  # Ensure axes is always iterable

        fig.suptitle(f"Top {num_plots} {title_prefix} Predictions", fontsize=16)

        for i, index in enumerate(indices):
            ax = axes[i]
            if is_online:
                branch_coords = x_test_plot[index]
                dt = branch_coords[1] - branch_coords[0]
                truth_coord = branch_coords[-1] + dt

                ax.plot(branch_coords, x_test[0][index, :-1], 'o--', color='purple', alpha=0.6,
                        label='Branch Input (u)')
                ax.plot(truth_coord, y_test[index], '*', markersize=12, color='blue', label='Ground Truth')
                ax.plot(truth_coord, y_pred[index], 'x', markersize=10, markeredgewidth=2, color=color,
                        label='Prediction')
            else:  # Offline
                trunk_coords = x_test[1][:, 0]
                ax.plot(trunk_coords, y_test[index], '-', color='blue', linewidth=2, label='Ground Truth')
                ax.plot(trunk_coords, y_pred[index], '--', color=color, linewidth=2, label='Prediction')

            ax.legend()
            ax.grid(True, linestyle='--')
            ax.set_title(f"Sample Index: {index} (L2 Error: {errors[index]:.2%})")

        axes[-1].set_xlabel("Time or Coordinate")
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)