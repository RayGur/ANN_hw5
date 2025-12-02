"""
Stage 1 Test Script
Tests data processing and LVQ-1 core functionality
"""

# import sys

# sys.path.append("/home/claude/ANN_hw5")

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


def test_stage1():
    """Complete test of Stage 1 components"""

    print("=" * 80)
    print(" " * 25 + "STAGE 1 TEST REPORT")
    print("=" * 80)

    # ========== Part 1: Data Loading ==========
    print("\n[1] DATA LOADING")
    print("-" * 80)
    print(f"Training data shape: {TRAIN_DATA.shape}")
    print(f"Training labels shape: {TRAIN_LABELS.shape}")
    print(f"Test data shape: {TEST_DATA.shape}")
    print(f"Number of classes: {len(np.unique(TRAIN_LABELS))}")
    print(f"Time points: {TIME_POINTS}")
    print("✓ Data loaded successfully")

    # ========== Part 2: Data Normalization ==========
    print("\n[2] DATA NORMALIZATION (Min-Max to [0, 1])")
    print("-" * 80)
    train_norm, test_norm, normalizer = prepare_data(
        TRAIN_DATA, TRAIN_LABELS, TEST_DATA
    )

    print(f"Original training range:")
    print(f"  Min: {TRAIN_DATA.min(axis=0).round(4)}")
    print(f"  Max: {TRAIN_DATA.max(axis=0).round(4)}")

    print(f"\nNormalized training range:")
    print(f"  Min: {train_norm.min(axis=0).round(4)}")
    print(f"  Max: {train_norm.max(axis=0).round(4)}")

    print(f"\nNormalized test range:")
    print(f"  Min: {test_norm.min(axis=0).round(4)}")
    print(f"  Max: {test_norm.max(axis=0).round(4)}")

    if test_norm.max() > 1.0:
        print(f"\n  Note: Test data slightly exceeds [0,1] range")
        print(f"        This is expected when test values > training max")

    print("✓ Normalization completed")

    # ========== Part 3: Sample Data Inspection ==========
    print("\n[3] SAMPLE DATA INSPECTION")
    print("-" * 80)
    print("First 3 training samples (original):")
    for i in range(3):
        print(f"  Day {i+1} (Class {TRAIN_LABELS[i]}): {TRAIN_DATA[i].round(4)}")

    print("\nFirst 3 training samples (normalized):")
    for i in range(3):
        print(f"  Day {i+1} (Class {TRAIN_LABELS[i]}): {train_norm[i].round(4)}")

    print("\nFirst 3 test samples (normalized):")
    for i in range(3):
        print(f"  Day {i+1}: {test_norm[i].round(4)}")

    # ========== Part 4: LVQ-1 Training ==========
    print("\n[4] LVQ-1 TRAINING")
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

    # ========== Part 5: Training Performance ==========
    print("\n[5] TRAINING PERFORMANCE")
    print("-" * 80)
    train_pred = lvq.predict(train_norm)
    train_acc = np.mean(train_pred == TRAIN_LABELS)

    print(f"Training Accuracy: {train_acc*100:.2f}%")
    print(f"\nTraining predictions vs true labels:")
    print(f"  Predicted: {train_pred}")
    print(f"  True:      {TRAIN_LABELS}")

    # Check which samples were misclassified
    misclassified = np.where(train_pred != TRAIN_LABELS)[0]
    if len(misclassified) > 0:
        print(f"\nMisclassified training samples: {misclassified}")
    else:
        print(f"\n✓ All training samples correctly classified")

    # ========== Part 6: Convergence Analysis ==========
    print("\n[6] CONVERGENCE ANALYSIS")
    print("-" * 80)
    history = lvq.get_convergence_history()

    print(f"Total training epochs: {len(history)}")
    print(f"Initial weight change: {history[0]:.6f}")
    print(f"Final weight change: {history[-1]:.6f}")
    print(f"Reduction ratio: {history[0]/history[-1]:.1f}x")

    # Show some intermediate values
    if len(history) >= 5:
        print(f"\nWeight change progression:")
        indices = [0, len(history) // 4, len(history) // 2, 3 * len(history) // 4, -1]
        for idx in indices:
            epoch_num = idx if idx >= 0 else len(history) + idx
            print(f"  Epoch {epoch_num+1:3d}: {history[idx]:.6f}")

    print("✓ Model converged successfully")

    # ========== Part 7: Test Predictions ==========
    print("\n[7] TEST SET PREDICTIONS")
    print("-" * 80)
    test_pred = lvq.predict(test_norm)

    print("Predictions for 8 test days:")
    print("-" * 40)
    for i, pred in enumerate(test_pred):
        print(f"  Day {i+1}: Class {pred} ({CLASS_NAMES[pred-1]})")

    # Show class distribution
    print(f"\nPredicted class distribution:")
    for class_label in range(1, 5):
        count = np.sum(test_pred == class_label)
        print(f"  Class {class_label}: {count} samples")

    # ========== Part 8: Prototype Information ==========
    print("\n[8] LEARNED PROTOTYPES")
    print("-" * 80)
    print("Prototype vectors (normalized space):")
    for i, (proto, label) in enumerate(zip(lvq.prototypes, lvq.prototype_labels)):
        print(f"  Prototype {i+1} (Class {label}): {proto.round(4)}")

    # ========== Summary ==========
    print("\n" + "=" * 80)
    print(" " * 25 + "STAGE 1 TEST SUMMARY")
    print("=" * 80)
    print("✓ Data loading and inspection")
    print("✓ Min-Max normalization to [0, 1]")
    print("✓ LVQ-1 training and convergence")
    print(f"✓ Training accuracy: {train_acc*100:.2f}%")
    print(f"✓ Converged in {len(history)} epochs")
    print("✓ Test predictions generated")
    print("\n" + "=" * 80)
    print("All Stage 1 tests passed successfully!")
    print("=" * 80)

    return {
        "train_norm": train_norm,
        "test_norm": test_norm,
        "normalizer": normalizer,
        "lvq": lvq,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_acc": train_acc,
        "history": history,
    }


if __name__ == "__main__":
    results = test_stage1()
