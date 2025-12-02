"""
Experiment 1: Basic LVQ-1 Prediction
Train LVQ-1 model and generate visualizations
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from data.power_data import (
    TRAIN_DATA,
    TRAIN_LABELS,
    TEST_DATA,
    TIME_POINTS,
    CLASS_NAMES,
)
from src.data_processing import prepare_data
from src.lvq_model import LVQ1Classifier
from src.visualization import (
    plot_convergence_curve,
    plot_all_pairwise_classifications,
    save_prediction_results,
    create_summary_table,
)


def run_experiment1(output_dir="output"):
    """
    Run Experiment 1: Basic LVQ-1 prediction with visualization

    Parameters:
    -----------
    output_dir : str
        Directory to save output files
    """
    print("=" * 80)
    print(" " * 20 + "EXPERIMENT 1: BASIC LVQ-1 PREDICTION")
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
    print(f"✓ Data normalized to [0, 1] range")

    # ========== Step 2: Train LVQ-1 Model ==========
    print("\n[Step 2] Training LVQ-1 model...")
    print("-" * 80)
    print("Configuration:")
    print("  - Prototypes per class: 1")
    print("  - Learning rate: 0.1")
    print("  - Max epochs: 500")
    print("  - Convergence threshold: 1e-4")
    print("  - Random seed: 42")
    print()

    lvq = LVQ1Classifier(
        n_prototypes=1,
        learning_rate=0.1,
        max_epochs=500,
        convergence_threshold=1e-4,
        random_state=42,
    )

    lvq.fit(train_norm, TRAIN_LABELS, verbose=True)

    # ========== Step 3: Evaluate Performance ==========
    print("\n[Step 3] Evaluating performance...")
    print("-" * 80)

    # Training accuracy
    train_pred = lvq.predict(train_norm)
    train_acc = np.mean(train_pred == TRAIN_LABELS)
    print(f"Training Accuracy: {train_acc*100:.2f}%")

    # Test predictions
    test_pred = lvq.predict(test_norm)
    print(f"\nTest Predictions:")
    for i, pred in enumerate(test_pred):
        print(f"  Day {i+1}: Class {pred} ({CLASS_NAMES[pred-1]})")

    # Convergence info
    history = lvq.get_convergence_history()
    print(f"\nConvergence Information:")
    print(f"  Epochs: {len(history)}")
    print(f"  Initial weight change: {history[0]:.6f}")
    print(f"  Final weight change: {history[-1]:.6f}")

    # ========== Step 4: Generate Visualizations ==========
    print("\n[Step 4] Generating visualizations...")
    print("-" * 80)

    # 4.1 Convergence curve
    print("\n4.1 Convergence curve...")
    conv_path = output_path / "convergence_curve.png"
    plot_convergence_curve(
        history,
        save_path=str(conv_path),
        title="LVQ-1 Convergence Curve (Learning Rate = 0.1)",
    )

    # 4.2 Pairwise classification plots
    print("\n4.2 Pairwise classification plots...")

    # Convert prototypes back to original scale for visualization
    prototypes_original = normalizer.inverse_transform(lvq.prototypes)

    plot_all_pairwise_classifications(
        TRAIN_DATA,
        TRAIN_LABELS,
        TEST_DATA,
        test_pred,
        prototypes_original,
        lvq.prototype_labels,
        TIME_POINTS,
        output_dir=str(output_path),
    )

    # 4.3 Save prediction results
    print("\n4.3 Saving prediction results...")
    pred_path = output_path / "prediction_results.txt"
    save_prediction_results(test_pred, str(pred_path))

    # 4.4 Save summary
    print("\n4.4 Generating summary report...")
    summary = create_summary_table(train_acc, test_pred, len(history), history[-1])

    summary_path = output_path / "experiment1_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved summary to: {summary_path}")

    # ========== Final Report ==========
    print("\n" + "=" * 80)
    print(" " * 25 + "EXPERIMENT 1 COMPLETED")
    print("=" * 80)
    print(f"\nOutput directory: {output_path}")
    print("\nGenerated files:")
    print("  ✓ convergence_curve.png")
    print("  ✓ classification_t1_vs_t2.png")
    print("  ✓ classification_t2_vs_t3.png")
    print("  ✓ classification_t3_vs_t4.png")
    print("  ✓ classification_t4_vs_t5.png")
    print("  ✓ classification_t5_vs_t6.png")
    print("  ✓ classification_t6_vs_t1.png")
    print("  ✓ prediction_results.txt")
    print("  ✓ experiment1_summary.txt")

    print("\n" + "=" * 80)
    print("Key Results:")
    print(f"  - Training Accuracy: {train_acc*100:.2f}%")
    print(f"  - Convergence Epochs: {len(history)}")
    print(f"  - Test Predictions: {test_pred}")
    print("=" * 80)

    return {
        "lvq": lvq,
        "train_acc": train_acc,
        "test_pred": test_pred,
        "history": history,
        "normalizer": normalizer,
    }


if __name__ == "__main__":
    results = run_experiment1()
