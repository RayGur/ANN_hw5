"""
LVQ-1 (Learning Vector Quantization) Implementation
Based on Kohonen's LVQ-1 algorithm
"""

import numpy as np


class LVQ1Classifier:
    """
    LVQ-1 Classifier for pattern classification

    Parameters:
    -----------
    n_prototypes : int
        Number of prototype vectors per class
    learning_rate : float, default=0.1
        Learning rate for weight updates
    max_epochs : int, default=500
        Maximum number of training epochs
    convergence_threshold : float, default=1e-4
        Threshold for convergence detection
    random_state : int or None, default=None
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_prototypes=1,
        learning_rate=0.1,
        max_epochs=500,
        convergence_threshold=1e-4,
        random_state=None,
    ):
        self.n_prototypes = n_prototypes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.convergence_threshold = convergence_threshold
        self.random_state = random_state

        # Will be set during training
        self.prototypes = None  # Prototype weight vectors
        self.prototype_labels = None  # Class labels for each prototype
        self.convergence_history = []  # Track weight changes per epoch
        self.n_classes = None
        self.n_features = None

        if random_state is not None:
            np.random.seed(random_state)

    def _initialize_prototypes(self, X, y):
        """
        Initialize prototype vectors by randomly selecting samples from each class

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Training data
        y : ndarray, shape (n_samples,)
            Training labels
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]

        prototypes = []
        prototype_labels = []

        # For each class, randomly select n_prototypes samples
        for class_label in np.unique(y):
            class_samples = X[y == class_label]

            # Randomly select samples from this class
            n_samples = min(self.n_prototypes, len(class_samples))
            indices = np.random.choice(
                len(class_samples), size=n_samples, replace=False
            )

            for idx in indices:
                prototypes.append(class_samples[idx].copy())
                prototype_labels.append(class_label)

        self.prototypes = np.array(prototypes)
        self.prototype_labels = np.array(prototype_labels)

        print(
            f"Initialized {len(self.prototypes)} prototypes for {self.n_classes} classes"
        )

    def _euclidean_distance(self, x, prototypes):
        """
        Calculate Euclidean distance between sample x and all prototypes

        Parameters:
        -----------
        x : ndarray, shape (n_features,)
            Input sample
        prototypes : ndarray, shape (n_prototypes, n_features)
            Prototype vectors

        Returns:
        --------
        distances : ndarray, shape (n_prototypes,)
            Distances from x to each prototype
        """
        # dist = sqrt(sum((x_i - w_i)^2))
        distances = np.sqrt(np.sum((prototypes - x) ** 2, axis=1))
        return distances

    def _find_winner(self, x):
        """
        Find the winner (closest) prototype for input sample x

        Parameters:
        -----------
        x : ndarray, shape (n_features,)
            Input sample

        Returns:
        --------
        winner_idx : int
            Index of the winning prototype
        """
        distances = self._euclidean_distance(x, self.prototypes)
        winner_idx = np.argmin(distances)
        return winner_idx

    def fit(self, X, y, verbose=True):
        """
        Train the LVQ-1 network

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Training data (normalized)
        y : ndarray, shape (n_samples,)
            Training labels
        verbose : bool, default=True
            Print training progress

        Returns:
        --------
        self : LVQ1Classifier
            Fitted classifier
        """
        # Initialize prototypes
        self._initialize_prototypes(X, y)

        # Training loop
        self.convergence_history = []

        for epoch in range(self.max_epochs):
            # Store prototypes before update
            prototypes_old = self.prototypes.copy()

            # Process all training samples
            for sample_idx in range(len(X)):
                x_k = X[sample_idx]
                desired_class = y[sample_idx]

                # Find winner neuron
                winner_idx = self._find_winner(x_k)
                winner_class = self.prototype_labels[winner_idx]

                # Update rule according to LVQ-1
                if winner_class == desired_class:
                    # Move winner closer to sample (correct classification)
                    self.prototypes[winner_idx] += self.learning_rate * (
                        x_k - self.prototypes[winner_idx]
                    )
                else:
                    # Move winner away from sample (incorrect classification)
                    self.prototypes[winner_idx] -= self.learning_rate * (
                        x_k - self.prototypes[winner_idx]
                    )

                # Normalize the updated prototype (as per PDF Step 7.1.4)
                norm = np.linalg.norm(self.prototypes[winner_idx])
                if norm > 0:
                    self.prototypes[winner_idx] /= norm

            # Calculate weight change (sum of differences)
            weight_change = np.sum(np.abs(self.prototypes - prototypes_old))
            self.convergence_history.append(weight_change)

            # Check convergence
            if weight_change < self.convergence_threshold:
                if verbose:
                    print(
                        f"Converged at epoch {epoch+1} (weight change: {weight_change:.6f})"
                    )
                break

            # Print progress
            if verbose and (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{self.max_epochs}, Weight change: {weight_change:.6f}"
                )

        if verbose and epoch == self.max_epochs - 1:
            print(
                f"Reached max epochs ({self.max_epochs}), final weight change: {weight_change:.6f}"
            )

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Test data (normalized)

        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels
        """
        if self.prototypes is None:
            raise ValueError("Model must be trained before prediction")

        predictions = []

        for x in X:
            # Find winner neuron
            winner_idx = self._find_winner(x)
            # Get the class of the winner
            predicted_class = self.prototype_labels[winner_idx]
            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, X, y):
        """
        Calculate classification accuracy

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Test data
        y : ndarray, shape (n_samples,)
            True labels

        Returns:
        --------
        accuracy : float
            Classification accuracy (0 to 1)
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def get_convergence_history(self):
        """
        Get the convergence history (weight changes per epoch)

        Returns:
        --------
        history : list
            List of weight changes for each epoch
        """
        return self.convergence_history


if __name__ == "__main__":
    # Test the LVQ-1 classifier
    import sys

    sys.path.append("/home/claude/ANN_hw5")

    from data.power_data import TRAIN_DATA, TRAIN_LABELS, TEST_DATA
    from src.data_processing import prepare_data

    print("=" * 60)
    print("Testing LVQ-1 Classifier")
    print("=" * 60)

    # Prepare data
    train_norm, test_norm, normalizer = prepare_data(
        TRAIN_DATA, TRAIN_LABELS, TEST_DATA
    )

    print(f"\nTraining data shape: {train_norm.shape}")
    print(f"Training labels shape: {TRAIN_LABELS.shape}")
    print(f"Test data shape: {test_norm.shape}")

    # Train LVQ-1
    print("\n" + "=" * 60)
    print("Training LVQ-1...")
    print("=" * 60)

    lvq = LVQ1Classifier(
        n_prototypes=1,
        learning_rate=0.1,
        max_epochs=500,
        convergence_threshold=1e-4,
        random_state=42,
    )

    lvq.fit(train_norm, TRAIN_LABELS, verbose=True)

    # Training accuracy
    train_pred = lvq.predict(train_norm)
    train_acc = np.mean(train_pred == TRAIN_LABELS)
    print(f"\nTraining Accuracy: {train_acc*100:.2f}%")

    # Test predictions
    test_pred = lvq.predict(test_norm)
    print(f"\nTest Predictions:")
    for i, pred in enumerate(test_pred):
        print(f"  Day {i+1}: Class {pred}")

    # Convergence history
    history = lvq.get_convergence_history()
    print(f"\nConvergence History:")
    print(f"  Total epochs: {len(history)}")
    print(f"  Initial weight change: {history[0]:.6f}")
    print(f"  Final weight change: {history[-1]:.6f}")

    print("\n" + "=" * 60)
    print("✓ LVQ-1 Classifier test completed!")
    print("=" * 60)
