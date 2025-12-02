"""
Stage 2 Test Script
Tests visualization and experiment 1 functionality
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from data.power_data import TRAIN_DATA, TRAIN_LABELS, TEST_DATA
from src.data_processing import prepare_data
from src.lvq_model import LVQ1Classifier
from src.visualization import plot_convergence_curve


def test_stage2():
    """Test Stage 2 components"""

    print("=" * 80)
    print(" " * 25 + "STAGE 2 TEST REPORT")
    print("=" * 80)

    # ========== Test 1: Visualization Module ==========
    print("\n[Test 1] Testing visualization module...")
    print("-" * 80)

    # Test convergence curve plotting
    test_history = [2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]

    fig, ax = plot_convergence_curve(
        test_history, save_path=None, title="Test Convergence Curve"
    )

    print("✓ Convergence curve plotting function works")
    plt.close()

    # ========== Test 2: Full Experiment 1 ==========
    print("\n[Test 2] Running full Experiment 1...")
    print("-" * 80)
    print("This will:")
    print("  - Train LVQ-1 model")
    print("  - Generate convergence curve")
    print("  - Generate 6 pairwise classification plots")
    print("  - Save prediction results")
    print("  - Create summary report")
    print()

    from experiments.exp1_basic_prediction import run_experiment1

    results = run_experiment1(output_dir="output")

    # ========== Verify Outputs ==========
    print("\n[Test 3] Verifying generated outputs...")
    print("-" * 80)

    output_dir = Path(project_root) / "output"

    expected_files = [
        "convergence_curve.png",
        "classification_t1_vs_t2.png",
        "classification_t2_vs_t3.png",
        "classification_t3_vs_t4.png",
        "classification_t4_vs_t5.png",
        "classification_t5_vs_t6.png",
        "classification_t6_vs_t1.png",
        "prediction_results.txt",
        "experiment1_summary.txt",
    ]

    all_files_exist = True
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {filename} ({size:,} bytes)")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            all_files_exist = False

    # ========== Verify Results Quality ==========
    print("\n[Test 4] Verifying results quality...")
    print("-" * 80)

    print(f"Training Accuracy: {results['train_acc']*100:.2f}%")
    if results["train_acc"] == 1.0:
        print("  ✓ Perfect training accuracy achieved")
    else:
        print("  ⚠ Training accuracy not perfect")

    print(f"\nConvergence Epochs: {len(results['history'])}")
    if len(results["history"]) < 100:
        print("  ✓ Model converged efficiently")
    else:
        print("  ⚠ Model took many epochs to converge")

    print(f"\nTest Predictions: {results['test_pred']}")

    # Check class distribution
    class_counts = [np.sum(results["test_pred"] == i) for i in range(1, 5)]
    print(f"Class distribution: {class_counts}")

    if all(c == 2 for c in class_counts):
        print("  ✓ Perfect class balance in predictions")
    else:
        print("  ⚠ Class imbalance detected")

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print(" " * 25 + "STAGE 2 TEST SUMMARY")
    print("=" * 80)

    if all_files_exist:
        print("✓ All visualization functions working")
        print("✓ All expected output files generated")
        print("✓ Experiment 1 completed successfully")
        print(f"✓ Training accuracy: {results['train_acc']*100:.2f}%")
        print(f"✓ Converged in {len(results['history'])} epochs")

        print("\n" + "=" * 80)
        print("All Stage 2 tests passed successfully!")
        print("=" * 80)

        return True
    else:
        print("✗ Some files missing")
        print("\n" + "=" * 80)
        print("Stage 2 tests FAILED!")
        print("=" * 80)

        return False


if __name__ == "__main__":
    success = test_stage2()
    sys.exit(0 if success else 1)
