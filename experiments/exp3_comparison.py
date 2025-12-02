"""
Experiment 3: LVQ vs K-means Comparison
Compare LVQ-1 and K-means clustering performance
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from data.power_data import TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TIME_POINTS
from src.data_processing import prepare_data
from src.lvq_model import LVQ1Classifier
from src.kmeans_baseline import KMeansClassifier


def plot_comparison_classification(
    X_train,
    y_train,
    X_test_lvq,
    y_pred_lvq,
    X_test_kmeans,
    y_pred_kmeans,
    prototypes_lvq,
    prototypes_kmeans,
    time_idx1,
    time_idx2,
    time_points,
    save_path=None,
):
    """
    Plot side-by-side comparison of LVQ and K-means classifications
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = ["red", "blue", "green", "black"]
    class_names = ["Class 1", "Class 2", "Class 3", "Class 4"]

    # Plot LVQ results
    for class_label in range(1, 5):
        mask = y_train == class_label
        ax1.scatter(
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

    for class_label in range(1, 5):
        mask = y_pred_lvq == class_label
        if np.sum(mask) > 0:
            ax1.scatter(
                X_test_lvq[mask, time_idx1],
                X_test_lvq[mask, time_idx2],
                c=colors[class_label - 1],
                marker="x",
                s=200,
                linewidth=3,
            )

    for proto in prototypes_lvq:
        ax1.scatter(
            proto[time_idx1],
            proto[time_idx2],
            c="gold",
            marker="P",
            s=300,
            edgecolors="black",
            linewidth=2,
        )

    ax1.set_xlabel(f"{time_points[time_idx1]}", fontsize=12)
    ax1.set_ylabel(f"{time_points[time_idx2]}", fontsize=12)
    ax1.set_title("LVQ-1 Classification", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot K-means results
    for class_label in range(1, 5):
        mask = y_train == class_label
        ax2.scatter(
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

    for class_label in range(1, 5):
        mask = y_pred_kmeans == class_label
        if np.sum(mask) > 0:
            ax2.scatter(
                X_test_kmeans[mask, time_idx1],
                X_test_kmeans[mask, time_idx2],
                c=colors[class_label - 1],
                marker="x",
                s=200,
                linewidth=3,
            )

    for center in prototypes_kmeans:
        ax2.scatter(
            center[time_idx1],
            center[time_idx2],
            c="gold",
            marker="*",
            s=400,
            edgecolors="black",
            linewidth=2,
        )

    ax2.set_xlabel(f"{time_points[time_idx1]}", fontsize=12)
    ax2.set_ylabel(f"{time_points[time_idx2]}", fontsize=12)
    ax2.set_title("K-means Classification", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to: {save_path}")

    return fig, (ax1, ax2)


def run_experiment3(output_dir="output"):
    """
    Run Experiment 3: Compare LVQ-1 with K-means

    Parameters:
    -----------
    output_dir : str
        Directory to save output files
    """
    print("=" * 80)
    print(" " * 20 + "EXPERIMENT 3: LVQ vs K-MEANS COMPARISON")
    print("=" * 80)

    # Create output directory
    output_path = Path(project_root) / output_dir
    output_path.mkdir(exist_ok=True)

    # ========== Step 1: Data Preparation ==========
    print("\n[Step 1] Preparing data...")
    print("-" * 80)

    train_norm, test_norm, normalizer = prepare_data(
        TRAIN_DATA, TRAIN_LABELS, TEST_DATA
    )

    print(f"✓ Training data: {train_norm.shape}")
    print(f"✓ Test data: {test_norm.shape}")

    # ========== Step 2: Train LVQ-1 ==========
    print("\n[Step 2] Training LVQ-1...")
    print("-" * 80)

    lvq = LVQ1Classifier(
        n_prototypes=1,
        learning_rate=0.1,
        max_epochs=500,
        convergence_threshold=1e-4,
        random_state=42,
    )

    lvq.fit(train_norm, TRAIN_LABELS, verbose=True)

    train_pred_lvq = lvq.predict(train_norm)
    train_acc_lvq = np.mean(train_pred_lvq == TRAIN_LABELS)
    test_pred_lvq = lvq.predict(test_norm)
    history_lvq = lvq.get_convergence_history()

    print(f"\nLVQ-1 Results:")
    print(f"  Training Accuracy: {train_acc_lvq*100:.2f}%")
    print(f"  Convergence Epochs: {len(history_lvq)}")
    print(f"  Test Predictions: {test_pred_lvq}")

    # ========== Step 3: Train K-means ==========
    print("\n[Step 3] Training K-means...")
    print("-" * 80)

    kmeans = KMeansClassifier(n_clusters=4, random_state=42)
    kmeans.fit(train_norm, TRAIN_LABELS, verbose=True)

    train_pred_kmeans = kmeans.predict(train_norm)
    train_acc_kmeans = np.mean(train_pred_kmeans == TRAIN_LABELS)
    test_pred_kmeans = kmeans.predict(test_norm)

    print(f"\nK-means Results:")
    print(f"  Training Accuracy: {train_acc_kmeans*100:.2f}%")
    print(f"  Test Predictions: {test_pred_kmeans}")

    # ========== Step 4: Generate Comparison Visualizations ==========
    print("\n[Step 4] Generating comparison visualizations...")
    print("-" * 80)

    # Convert prototypes back to original scale
    prototypes_lvq_orig = normalizer.inverse_transform(lvq.prototypes)
    prototypes_kmeans_orig = normalizer.inverse_transform(kmeans.get_cluster_centers())

    # Plot comparison for time points 1 vs 2
    comp_path = output_path / "comparison_classification_t1_vs_t2.png"
    plot_comparison_classification(
        TRAIN_DATA,
        TRAIN_LABELS,
        TEST_DATA,
        test_pred_lvq,
        TEST_DATA,
        test_pred_kmeans,
        prototypes_lvq_orig,
        prototypes_kmeans_orig,
        0,
        1,
        TIME_POINTS,
        save_path=str(comp_path),
    )
    plt.close()

    # Plot comparison for time points 3 vs 4
    comp_path2 = output_path / "comparison_classification_t3_vs_t4.png"
    plot_comparison_classification(
        TRAIN_DATA,
        TRAIN_LABELS,
        TEST_DATA,
        test_pred_lvq,
        TEST_DATA,
        test_pred_kmeans,
        prototypes_lvq_orig,
        prototypes_kmeans_orig,
        2,
        3,
        TIME_POINTS,
        save_path=str(comp_path2),
    )
    plt.close()

    # ========== Step 5: Generate Comparison Report ==========
    print("\n[Step 5] Generating comparison report...")
    print("-" * 80)

    report_path = output_path / "comparison_results.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LVQ-1 vs K-means Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("Training Performance:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<15} {'Train Accuracy':<20} {'Additional Info'}\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'LVQ-1':<15} {train_acc_lvq*100:<19.2f}% "
            f"Converged in {len(history_lvq)} epochs\n"
        )
        f.write(
            f"{'K-means':<15} {train_acc_kmeans*100:<19.2f}% "
            f"Converged in {kmeans.kmeans.n_iter_} iterations\n"
        )

        f.write("\n" + "=" * 80 + "\n")
        f.write("Test Predictions:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Day':<8} {'LVQ-1':<10} {'K-means':<10} {'Match'}\n")
        f.write("-" * 80 + "\n")

        matches = 0
        for i, (pred_lvq, pred_kmeans) in enumerate(
            zip(test_pred_lvq, test_pred_kmeans)
        ):
            match = "YES" if pred_lvq == pred_kmeans else "NO"
            if pred_lvq == pred_kmeans:
                matches += 1
            f.write(f"{i+1:<8} {pred_lvq:<10} {pred_kmeans:<10} {match}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write(f"Prediction Agreement: {matches}/8 ({matches/8*100:.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Class Distribution in Test Predictions:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<10} {'LVQ-1':<10} {'K-means'}\n")
        f.write("-" * 80 + "\n")

        for class_label in range(1, 5):
            count_lvq = np.sum(test_pred_lvq == class_label)
            count_kmeans = np.sum(test_pred_kmeans == class_label)
            f.write(f"{class_label:<10} {count_lvq:<10} {count_kmeans}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Key Observations:\n")
        f.write("-" * 80 + "\n")

        if train_acc_lvq > train_acc_kmeans:
            f.write("• LVQ-1 achieved higher training accuracy\n")
        elif train_acc_kmeans > train_acc_lvq:
            f.write("• K-means achieved higher training accuracy\n")
        else:
            f.write("• Both methods achieved equal training accuracy\n")

        if matches == 8:
            f.write("• Both methods made identical predictions on all test samples\n")
        elif matches >= 6:
            f.write("• Both methods showed high agreement on test predictions\n")
        else:
            f.write("• Methods showed different prediction patterns\n")

        # Check class balance
        lvq_counts = [np.sum(test_pred_lvq == i) for i in range(1, 5)]
        kmeans_counts = [np.sum(test_pred_kmeans == i) for i in range(1, 5)]

        if all(c == 2 for c in lvq_counts):
            f.write("• LVQ-1 predictions are perfectly balanced across classes\n")
        if all(c == 2 for c in kmeans_counts):
            f.write("• K-means predictions are perfectly balanced across classes\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Saved comparison report to: {report_path}")

    # ========== Final Report ==========
    print("\n" + "=" * 80)
    print(" " * 25 + "EXPERIMENT 3 COMPLETED")
    print("=" * 80)
    print(f"\nOutput directory: {output_path}")
    print("\nGenerated files:")
    print("  ✓ comparison_classification_t1_vs_t2.png")
    print("  ✓ comparison_classification_t3_vs_t4.png")
    print("  ✓ comparison_results.txt")

    print("\n" + "=" * 80)
    print("Comparison Summary:")
    print("-" * 80)
    print(
        f"LVQ-1:   {train_acc_lvq*100:5.2f}% train acc, "
        f"{len(history_lvq):3d} epochs"
    )
    print(
        f"K-means: {train_acc_kmeans*100:5.2f}% train acc, "
        f"{kmeans.kmeans.n_iter_:3d} iterations"
    )
    print(f"\nPrediction agreement: {matches}/8 ({matches/8*100:.1f}%)")
    print("=" * 80)

    return {
        "lvq": lvq,
        "kmeans": kmeans,
        "train_acc_lvq": train_acc_lvq,
        "train_acc_kmeans": train_acc_kmeans,
        "test_pred_lvq": test_pred_lvq,
        "test_pred_kmeans": test_pred_kmeans,
        "matches": matches,
    }


if __name__ == "__main__":
    results = run_experiment3()
