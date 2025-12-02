"""
Experiment 2: Learning Rate Impact Analysis
Compare different learning rates and their effect on convergence
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
from src.visualization import plot_learning_rate_comparison


def run_experiment2(
    learning_rates=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], output_dir="output"
):
    """
    Run Experiment 2: Test different learning rates

    Parameters:
    -----------
    learning_rates : list
        List of learning rates to test
    output_dir : str
        Directory to save output files
    """
    print("=" * 80)
    print(" " * 20 + "EXPERIMENT 2: LEARNING RATE IMPACT ANALYSIS")
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

    # ========== Step 2: Train with Different Learning Rates ==========
    print("\n[Step 2] Training with different learning rates...")
    print("-" * 80)
    print(f"Testing learning rates: {learning_rates}")
    print()

    results = {}

    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"Training with Learning Rate = {lr}")
        print("=" * 60)

        lvq = LVQ1Classifier(
            n_prototypes=1,
            learning_rate=lr,
            max_epochs=500,
            convergence_threshold=1e-4,
            random_state=42,
        )

        lvq.fit(train_norm, TRAIN_LABELS, verbose=False)

        # Evaluate
        train_pred = lvq.predict(train_norm)
        train_acc = np.mean(train_pred == TRAIN_LABELS)
        test_pred = lvq.predict(test_norm)
        history = lvq.get_convergence_history()

        results[lr] = {
            "lvq": lvq,
            "train_acc": train_acc,
            "test_pred": test_pred,
            "history": history,
            "n_epochs": len(history),
        }

        print(f"  Training Accuracy: {train_acc*100:.2f}%")
        print(f"  Convergence Epochs: {len(history)}")
        print(f"  Final Weight Change: {history[-1]:.6f}")
        print(f"  Test Predictions: {test_pred}")

    # ========== Step 3: Generate Comparison Visualizations ==========
    print("\n" + "=" * 80)
    print("[Step 3] Generating comparison visualizations...")
    print("-" * 80)

    # 3.1 Convergence curves comparison
    print("\n3.1 Convergence curves comparison...")
    lr_histories = {lr: res["history"] for lr, res in results.items()}

    comp_path = output_path / "lr_comparison_convergence.png"
    plot_learning_rate_comparison(lr_histories, save_path=str(comp_path))

    # ========== Step 4: Generate Comparison Table ==========
    print("\n[Step 4] Generating comparison table...")
    print("-" * 80)

    table_path = output_path / "lr_comparison_table.txt"

    with open(table_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Learning Rate Comparison Results\n")
        f.write("=" * 80 + "\n\n")

        # Header
        f.write(
            f"{'LR':<8} {'Epochs':<10} {'Train Acc':<12} {'Final Weight':<15} {'Test Pred'}\n"
        )
        f.write("-" * 80 + "\n")

        # Data rows
        for lr in sorted(results.keys()):
            res = results[lr]
            f.write(
                f"{lr:<8.2f} {res['n_epochs']:<10} {res['train_acc']*100:<11.2f}% "
                f"{res['history'][-1]:<15.6f} {str(res['test_pred'])}\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write("Observations:\n")
        f.write("-" * 80 + "\n")

        # Find best/worst performers
        best_lr = min(results.keys(), key=lambda lr: results[lr]["n_epochs"])
        worst_lr = max(results.keys(), key=lambda lr: results[lr]["n_epochs"])

        f.write(
            f"• Fastest convergence: LR = {best_lr} ({results[best_lr]['n_epochs']} epochs)\n"
        )
        f.write(
            f"• Slowest convergence: LR = {worst_lr} ({results[worst_lr]['n_epochs']} epochs)\n"
        )

        # Check if all achieve same accuracy
        all_accs = [res["train_acc"] for res in results.values()]
        if all(acc == all_accs[0] for acc in all_accs):
            f.write(
                f"• All learning rates achieved {all_accs[0]*100:.2f}% training accuracy\n"
            )
        else:
            f.write(f"• Training accuracy varies across learning rates\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Saved comparison table to: {table_path}")

    # ========== Step 5: Display Summary ==========
    print("\n" + "=" * 80)
    print(" " * 25 + "EXPERIMENT 2 COMPLETED")
    print("=" * 80)
    print(f"\nOutput directory: {output_path}")
    print("\nGenerated files:")
    print("  ✓ lr_comparison_convergence.png")
    print("  ✓ lr_comparison_table.txt")

    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("-" * 80)

    for lr in sorted(results.keys()):
        res = results[lr]
        print(
            f"LR = {lr:4.2f}: {res['n_epochs']:3d} epochs, "
            f"{res['train_acc']*100:5.2f}% accuracy, "
            f"final Δw = {res['history'][-1]:.6f}"
        )

    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_experiment2()
