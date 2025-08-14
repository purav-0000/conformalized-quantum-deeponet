import deepxde as dde
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

class LRLogger(dde.callbacks.Callback):
    """A callback to print the learning rate."""

    def __init__(self, display_every=1_000):
        super().__init__()
        # Print epoch every 'reps'
        self.display_every = display_every

    def on_epoch_end(self):
        current_epoch = self.model.train_state.epoch

        if current_epoch % self.display_every == 0:
            current_lr = self.model.opt.param_groups[0]['lr']
            print(f"└─> [LR at Epoch {current_epoch}]: {current_lr:.6f}")


def create_decay_and_hold_scheduler(initial_lr: float, gamma: float, min_lr: float):
    """
    Lambda Scheduler that returns a multiplicative factor
    """
    # Calculate the minimum multiplicative factor that corresponds to the minimum LR
    min_factor = min_lr / initial_lr

    def scheduler(step):
        # Calculate the decay factor for the current step
        decay_factor = gamma ** step

        # Return the larger of the two factors: the decayed factor or the minimum factor
        return max(decay_factor, min_factor)

    return scheduler


def model_input_plotting(x_train, y_train: np.ndarray, model_dir: Path, x_train_plot: np.ndarray):
    """Check data being fed to model"""

    for i in range(3):
        index = np.random.randint(len(x_train[0]))
        plt.figure(figsize=(10, 6))

        # If online, create a plot with discretizations visible
        if y_train.shape[1] == 1:

            # -1 because last parameter is the augmented feature that represents norm
            plt.plot(x_train_plot[index], x_train[0][index, :-1], 'mo--', label=f'Branch input')

            # Identify time step difference
            dt = x_train_plot[index][1] - x_train_plot[index][0]
            plt.plot(x_train_plot[index][-1] + dt, y_train[index], 'b*', label='Ground truth')
            plt.legend()
            plt.grid(True)
            plt.title(f"Last component for branch: {x_train[0][index, -1]}")
            plt.savefig(f'{model_dir}/training_plots/input{i}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Line plot
        else:
            # -1 because last parameter is the augmented feature that represents norm
            plt.plot(x_train_plot, x_train[0][index, :-1], 'm--', label=f'Branch input')
            plt.title(f"Last component for branch: {x_train[0][index, -1]}")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{model_dir}/training_plots/input{i}_branch.png', dpi=300, bbox_inches='tight')
            plt.close()

            plt.plot(x_train[1][:, 0], y_train[index], 'b-', label='Ground truth')
            plt.title(f"(Sample Index: {index})")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{model_dir}/training_plots/input{i}_trunk.png', dpi=300, bbox_inches='tight')
            plt.close()


def model_output_plotting(model: dde.Model, x_test: np.ndarray, y_test: np.ndarray, model_dir: Path,
                          x_test_plot: np.ndarray):
    """Check the outputs of the model"""

    with torch.no_grad():
        # 1. Get predictions for the entire test set
        y_pred = model.predict(x_test)

        # 2. Calculate L2 relative error for each sample
        errors = (np.linalg.norm(y_pred - y_test, axis=1) / np.linalg.norm(y_test, axis=1))

        # 3. Plot the histogram of errors
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.75)
        plt.title("Distribution of L2 Relative Errors on Test Set")
        plt.xlabel("L2 Relative Error")
        plt.ylabel("Number of Samples")
        plt.grid(True)

        plt.savefig(f"{model_dir}/training_plots/error_hist.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Find and plot the worst and best predictions
        sorted_indices = np.argsort(errors)
        worst_indices = sorted_indices[-3:]  # Top 3 worst
        best_indices = sorted_indices[:3]  # Top 3 best

        logging.info(f"Worst L2 errors: {errors[worst_indices]}")
        logging.info(f"Best L2 errors: {errors[best_indices]}")

        # Plot worst predictions
        for i, index in enumerate(worst_indices):

            # Plot differently if online
            if y_test.shape[1] == 1:
                plt.figure(figsize=(10, 6))
                plt.plot(x_test_plot[index], x_test[0][index, :-1], 'mo--', label=f'Branch input')
                dt = x_test_plot[index][1] - x_test_plot[index][0]
                plt.plot(x_test_plot[index][-1] + dt, y_test[index], 'b*', label='Ground truth')
                plt.plot(x_test_plot[index][-1] + dt, y_pred[index], 'rx', label='Prediction')
            else:
                plt.figure(figsize=(10, 6))
                plt.plot(x_test[1][:, 0], y_test[index], 'b-', label=f'Ground Truth (Error: {errors[index]:.2%})')
                plt.plot(x_test[1][:, 0], y_pred[index], 'r--', label='Prediction')

                # Uncomment if branch input is desired
                # plt.plot(x_test_plot, x_test[0][index, :-1], 'm--', label=f'Branch input')
            plt.title(f"Worst Prediction #{i + 1} (Sample Index: {index})\n "
                      f"Last component for branch: {x_test[0][index, -1]}")
            plt.legend()
            plt.grid(True)

            plt.savefig(f"{model_dir}/training_plots/worst_pred_{index}.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Plot best predictions
        for i, index in enumerate(best_indices):
            # Plot differently if online
            if y_test.shape[1] == 1:
                plt.figure(figsize=(10, 6))
                plt.plot(x_test_plot[index], x_test[0][index, :-1], 'mo--', label=f'Branch input')
                dt = x_test_plot[index][1] - x_test_plot[index][0]
                plt.plot(x_test_plot[index][-1] + dt, y_test[index], 'b*', label='Ground truth')
                plt.plot(x_test_plot[index][-1] + dt, y_pred[index], 'gx', label='Prediction')
            else:
                plt.figure(figsize=(10, 6))
                plt.plot(x_test[1][:, 0], y_test[index], 'b-', label=f'Ground Truth (Error: {errors[index]:.2%})')
                plt.plot(x_test[1][:, 0], y_pred[index], 'g--', label='Prediction')

                # Uncomment if branch input is desired
                # plt.plot(x_test_plot, x_test[0][index, :-1], 'm--', label=f'Branch input')
            plt.title(f"Best Prediction #{i + 1} (Sample Index: {index})\n"
                      f"Last component for branch: {x_test[0][index, -1]}")
            plt.legend()
            plt.grid(True)

            plt.savefig(f"{model_dir}/training_plots/best_pred_{index}.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Plot 3 random samples
        for i in range(3):
            index = np.random.randint(len(x_test[0]))

            # Plot differently if online
            if y_test.shape[1] == 1:
                plt.figure(figsize=(10, 6))
                plt.plot(x_test_plot[index], x_test[0][index, :-1], 'mo--', label=f'Branch input')
                dt = x_test_plot[index][1] - x_test_plot[index][0]
                plt.plot(x_test_plot[index][-1] + dt, y_test[index], 'b*', label='Ground truth')
                plt.plot(x_test_plot[index][-1] + dt, y_pred[index], 'kx', label='Prediction')
            else:
                plt.figure(figsize=(10, 6))
                plt.plot(x_test[1][:, 0], y_test[index], 'b-', label=f'Ground Truth (Error: {errors[index]:.2%})')
                plt.plot(x_test[1][:, 0], y_pred[index], 'k--', label='Prediction')

                # Uncomment if branch input is desired
                # plt.plot(x_test_plot, x_test[0][index, :-1], 'm--', label=f'Branch input')
            plt.title(f"Random Prediction #{i + 1} (Sample Index: {index})\n"
                      f"Last component for branch: {x_test[0][index, -1]}")
            plt.legend()
            plt.grid(True)

            plt.savefig(f"{model_dir}/training_plots/random_pred_{index}.png", dpi=300, bbox_inches='tight')
            plt.close()
