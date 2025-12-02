"""
Visualization Module
Provides plotting functions for LVQ analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


def plot_convergence_curve(history, save_path=None, title="LVQ-1 Convergence Curve"):
    """
    Plot convergence curve showing weight changes over epochs

    Parameters:
    -----------
    history : list or array
        Weight change values per epoch
    save_path : str, optional
        Path to save the figure
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history) + 1)
    ax.plot(epochs, history, "b-", linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Sum of Weight Differences", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add convergence threshold line if visible
    if len(history) > 0:
        threshold = 1e-4
        if max(history) > threshold:
            ax.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold ({threshold})",
            )
            ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved convergence curve to: {save_path}")

    return fig, ax


def plot_pairwise_classification(
    X_train,
    y_train,
    X_test,
    y_pred_test,
    prototypes,
    prototype_labels,
    time_idx1,
    time_idx2,
    time_points,
    save_path=None,
):
    """
    Plot classification results for two time dimensions (similar to slide4)

    Parameters:
    -----------
    X_train : ndarray, shape (n_train, n_features)
        Training data (original scale, not normalized)
    y_train : ndarray, shape (n_train,)
        Training labels
    X_test : ndarray, shape (n_test, n_features)
        Test data (original scale)
    y_pred_test : ndarray, shape (n_test,)
        Predicted labels for test data
    prototypes : ndarray, shape (n_prototypes, n_features)
        Prototype vectors (original scale)
    prototype_labels : ndarray, shape (n_prototypes,)
        Labels for prototypes
    time_idx1 : int
        First time dimension index
    time_idx2 : int
        Second time dimension index
    time_points : list
        Names of time points
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define colors for 4 classes
    colors = ["red", "blue", "green", "black"]
    class_names = ["Class 1", "Class 2", "Class 3", "Class 4"]

    # Plot training samples
    for class_label in range(1, 5):
        mask = y_train == class_label
        ax.scatter(
            X_train[mask, time_idx1],
            X_train[mask, time_idx2],
            c=colors[class_label - 1],
            marker="o",
            s=100,
            alpha=0.6,
            edgecolors="black",
            linewidth=1,
            label=class_names[class_label - 1],
        )

    # Plot test samples with predicted labels
    for class_label in range(1, 5):
        mask = y_pred_test == class_label
        if np.sum(mask) > 0:
            ax.scatter(
                X_test[mask, time_idx1],
                X_test[mask, time_idx2],
                c=colors[class_label - 1],
                marker="x",
                s=200,
                linewidth=3,
                alpha=0.9,
            )

    # Plot prototypes
    for i, (proto, label) in enumerate(zip(prototypes, prototype_labels)):
        ax.scatter(
            proto[time_idx1],
            proto[time_idx2],
            c=colors[int(label) - 1],
            marker="P",
            s=300,
            edgecolors="gold",
            linewidth=2,
            label="_nolegend_",
        )

    # Labels and title
    ax.set_xlabel(f"X{time_idx1+1} ({time_points[time_idx1]})", fontsize=12)
    ax.set_ylabel(f"X{time_idx2+1} ({time_points[time_idx2]})", fontsize=12)

    title = f"{time_idx1+1} vs {time_idx2+1}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=10,
            label=class_names[i],
            markeredgecolor="black",
        )
        for i in range(4)
    ]
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            markersize=10,
            label="Test samples",
            linestyle="None",
            linewidth=3,
        )
    )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="P",
            color="gold",
            markerfacecolor="gray",
            markersize=12,
            label="Prototypes",
            markeredgewidth=2,
        )
    )

    ax.legend(handles=legend_elements, loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved classification plot to: {save_path}")

    return fig, ax


def plot_all_pairwise_classifications(
    X_train,
    y_train,
    X_test,
    y_pred_test,
    prototypes,
    prototype_labels,
    time_points,
    output_dir,
):
    """
    Generate all 6 pairwise classification plots
    Pairs: (1,2), (2,3), (3,4), (4,5), (5,6), (6,1)

    Parameters:
    -----------
    X_train : ndarray
        Training data (original scale)
    y_train : ndarray
        Training labels
    X_test : ndarray
        Test data (original scale)
    y_pred_test : ndarray
        Predicted test labels
    prototypes : ndarray
        Prototype vectors (original scale)
    prototype_labels : ndarray
        Prototype labels
    time_points : list
        Time point names
    output_dir : str
        Directory to save plots

    Returns:
    --------
    fig_paths : list
        List of saved figure paths
    """
    # Define pairs: (1,2), (2,3), (3,4), (4,5), (5,6), (6,1)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

    fig_paths = []

    print("\nGenerating pairwise classification plots...")
    print("-" * 60)

    for idx1, idx2 in pairs:
        filename = f"classification_t{idx1+1}_vs_t{idx2+1}.png"
        save_path = os.path.join(output_dir, filename)

        plot_pairwise_classification(
            X_train,
            y_train,
            X_test,
            y_pred_test,
            prototypes,
            prototype_labels,
            idx1,
            idx2,
            time_points,
            save_path=save_path,
        )

        fig_paths.append(save_path)
        plt.close()

    print("-" * 60)
    print(f"✓ Generated {len(fig_paths)} classification plots")

    return fig_paths


def save_prediction_results(test_predictions, output_path):
    """
    Save test predictions to a text file

    Parameters:
    -----------
    test_predictions : ndarray
        Predicted class labels for test set
    output_path : str
        Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("LVQ-1 Test Set Predictions\n")
        f.write("=" * 60 + "\n\n")

        f.write("Predictions for 8 test days:\n")
        f.write("-" * 40 + "\n")

        for i, pred in enumerate(test_predictions):
            f.write(f"Day {i+1}: Class {pred}\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("Class distribution:\n")

        for class_label in range(1, 5):
            count = np.sum(test_predictions == class_label)
            f.write(f"  Class {class_label}: {count} samples\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Saved prediction results to: {output_path}")


def plot_learning_rate_comparison(lr_results, save_path=None):
    """
    Plot convergence curves for different learning rates

    Parameters:
    -----------
    lr_results : dict
        Dictionary with learning rates as keys and convergence histories as values
        Example: {0.1: [2.5, 1.2, 0.5, ...], 0.2: [...], ...}
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lr_results)))

    for (lr, history), color in zip(sorted(lr_results.items()), colors):
        epochs = range(1, len(history) + 1)
        ax.plot(
            epochs,
            history,
            linewidth=2,
            marker="o",
            markersize=3,
            label=f"LR = {lr}",
            color=color,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Sum of Weight Differences", fontsize=12)
    ax.set_title(
        "Learning Rate Comparison - Convergence Curves", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Log scale for better visibility

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved learning rate comparison to: {save_path}")

    return fig, ax


def create_summary_table(train_acc, test_predictions, n_epochs, final_weight_change):
    """
    Create a summary text table of results

    Parameters:
    -----------
    train_acc : float
        Training accuracy
    test_predictions : ndarray
        Test predictions
    n_epochs : int
        Number of epochs until convergence
    final_weight_change : float
        Final weight change value

    Returns:
    --------
    summary : str
        Formatted summary text
    """
    summary = []
    summary.append("=" * 60)
    summary.append("LVQ-1 Training Summary")
    summary.append("=" * 60)
    summary.append(f"\nTraining Accuracy: {train_acc*100:.2f}%")
    summary.append(f"Convergence Epochs: {n_epochs}")
    summary.append(f"Final Weight Change: {final_weight_change:.6f}")

    summary.append("\n" + "-" * 60)
    summary.append("Test Predictions:")
    summary.append("-" * 60)

    for i, pred in enumerate(test_predictions):
        summary.append(f"  Day {i+1}: Class {pred}")

    summary.append("\n" + "-" * 60)
    summary.append("Class Distribution in Test Set:")
    for class_label in range(1, 5):
        count = np.sum(test_predictions == class_label)
        summary.append(f"  Class {class_label}: {count} samples")

    summary.append("\n" + "=" * 60)

    return "\n".join(summary)


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization module...")

    # Test convergence curve
    test_history = [2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
    fig, ax = plot_convergence_curve(
        test_history, save_path=None, title="Test Convergence Curve"
    )
    plt.show()

    print("✓ Visualization module test completed")
