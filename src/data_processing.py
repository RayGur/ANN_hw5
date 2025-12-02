"""
Data Processing Module
Handles normalization and data preparation
"""

import numpy as np


class MinMaxNormalizer:
    """
    Min-Max Normalization: scales data to [0, 1] range
    Formula: x_norm = (x - x_min) / (x_max - x_min)
    """

    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        self.fitted = False

    def fit(self, X):
        """
        Compute min and max values for each feature

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Training data
        """
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)
        self.fitted = True

    def transform(self, X):
        """
        Apply normalization to data

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Data to normalize

        Returns:
        --------
        X_norm : ndarray, shape (n_samples, n_features)
            Normalized data in [0, 1] range
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        # Avoid division by zero
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0

        X_norm = (X - self.min_vals) / range_vals
        return X_norm

    def fit_transform(self, X):
        """
        Fit and transform in one step

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Training data

        Returns:
        --------
        X_norm : ndarray, shape (n_samples, n_features)
            Normalized data
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_norm):
        """
        Convert normalized data back to original scale

        Parameters:
        -----------
        X_norm : ndarray, shape (n_samples, n_features)
            Normalized data

        Returns:
        --------
        X : ndarray, shape (n_samples, n_features)
            Data in original scale
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0

        X = X_norm * range_vals + self.min_vals
        return X


def prepare_data(train_data, train_labels, test_data):
    """
    Prepare and normalize data for LVQ training

    Parameters:
    -----------
    train_data : ndarray, shape (n_train, n_features)
        Training samples
    train_labels : ndarray, shape (n_train,)
        Training labels
    test_data : ndarray, shape (n_test, n_features)
        Test samples

    Returns:
    --------
    train_norm : ndarray
        Normalized training data
    test_norm : ndarray
        Normalized test data
    normalizer : MinMaxNormalizer
        Fitted normalizer object
    """
    # Initialize normalizer
    normalizer = MinMaxNormalizer()

    # Fit on training data and transform
    train_norm = normalizer.fit_transform(train_data)

    # Transform test data using training statistics
    test_norm = normalizer.transform(test_data)

    return train_norm, test_norm, normalizer


if __name__ == "__main__":
    # Test the normalizer
    from data.power_data import TRAIN_DATA, TRAIN_LABELS, TEST_DATA

    print("=" * 60)
    print("Testing Data Processing Module")
    print("=" * 60)

    # Test normalization
    train_norm, test_norm, normalizer = prepare_data(
        TRAIN_DATA, TRAIN_LABELS, TEST_DATA
    )

    print(f"\nOriginal Training Data Shape: {TRAIN_DATA.shape}")
    print(f"Normalized Training Data Shape: {train_norm.shape}")
    print(f"\nOriginal Test Data Shape: {TEST_DATA.shape}")
    print(f"Normalized Test Data Shape: {test_norm.shape}")

    print(f"\nOriginal Training Data Range:")
    print(f"  Min: {TRAIN_DATA.min(axis=0)}")
    print(f"  Max: {TRAIN_DATA.max(axis=0)}")

    print(f"\nNormalized Training Data Range:")
    print(f"  Min: {train_norm.min(axis=0)}")
    print(f"  Max: {train_norm.max(axis=0)}")

    print(f"\nNormalized Test Data Range:")
    print(f"  Min: {test_norm.min(axis=0)}")
    print(f"  Max: {test_norm.max(axis=0)}")

    print(f"\nFirst 3 Original Training Samples:")
    print(TRAIN_DATA[:3])

    print(f"\nFirst 3 Normalized Training Samples:")
    print(train_norm[:3])

    print(f"\nFirst 3 Normalized Test Samples:")
    print(test_norm[:3])

    # Verify training data is in [0, 1]
    assert (
        train_norm.min() >= 0.0 and train_norm.max() <= 1.0
    ), "Training data not in [0,1]"
    print("\n✓ Training data normalized to [0, 1]")

    # Test data may slightly exceed [0,1] if values are outside training range
    if test_norm.min() < 0.0 or test_norm.max() > 1.0:
        print(
            f"⚠ Test data slightly outside [0,1]: [{test_norm.min():.4f}, {test_norm.max():.4f}]"
        )
        print("  (This is expected when test values exceed training range)")
    else:
        print("✓ Test data also in [0, 1]")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
