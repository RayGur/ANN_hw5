"""
K-means Baseline Implementation
Using scikit-learn's KMeans for comparison with LVQ
"""

import numpy as np
from sklearn.cluster import KMeans


class KMeansClassifier:
    """
    K-means wrapper for classification task

    Parameters:
    -----------
    n_clusters : int, default=4
        Number of clusters
    max_iter : int, default=300
        Maximum number of iterations
    random_state : int or None, default=None
        Random seed
    """

    def __init__(self, n_clusters=4, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        self.kmeans = None
        self.cluster_labels = None  # Map cluster ID to class label
        self.convergence_history = []

    def fit(self, X, y, verbose=True):
        """
        Fit K-means and assign cluster labels based on majority voting

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Training data
        y : ndarray, shape (n_samples,)
            Training labels (for label assignment)
        verbose : bool
            Print training info

        Returns:
        --------
        self : KMeansClassifier
        """
        if verbose:
            print(f"Training K-means with {self.n_clusters} clusters...")

        # Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=10,
        )

        self.kmeans.fit(X)

        # Assign labels to clusters based on majority voting
        cluster_assignments = self.kmeans.predict(X)
        self.cluster_labels = {}

        for cluster_id in range(self.n_clusters):
            # Find samples in this cluster
            mask = cluster_assignments == cluster_id
            if np.sum(mask) > 0:
                # Assign the most common label in this cluster
                labels_in_cluster = y[mask]
                unique, counts = np.unique(labels_in_cluster, return_counts=True)
                self.cluster_labels[cluster_id] = unique[np.argmax(counts)]
            else:
                self.cluster_labels[cluster_id] = 1  # Default

        if verbose:
            print(f"K-means converged in {self.kmeans.n_iter_} iterations")
            print(f"Cluster label assignments: {self.cluster_labels}")

        # Simulate convergence history (K-means doesn't provide this directly)
        # We'll use inertia (sum of squared distances) as a proxy
        self.convergence_history = [self.kmeans.inertia_]

        return self

    def predict(self, X):
        """
        Predict class labels for samples

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Test data

        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels
        """
        if self.kmeans is None:
            raise ValueError("Model must be fitted before prediction")

        # Get cluster assignments
        cluster_assignments = self.kmeans.predict(X)

        # Map to class labels
        predictions = np.array([self.cluster_labels[c] for c in cluster_assignments])

        return predictions

    def score(self, X, y):
        """
        Calculate classification accuracy

        Parameters:
        -----------
        X : ndarray
            Test data
        y : ndarray
            True labels

        Returns:
        --------
        accuracy : float
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_cluster_centers(self):
        """
        Get cluster centers

        Returns:
        --------
        centers : ndarray
            Cluster centers
        """
        return self.kmeans.cluster_centers_

    def get_convergence_history(self):
        """
        Get convergence history (inertia value)

        Returns:
        --------
        history : list
        """
        return self.convergence_history


if __name__ == "__main__":
    # Test K-means classifier
    import sys

    sys.path.append("/home/claude/ANN_hw5")

    from data.power_data import TRAIN_DATA, TRAIN_LABELS, TEST_DATA
    from src.data_processing import prepare_data

    print("=" * 60)
    print("Testing K-means Classifier")
    print("=" * 60)

    # Prepare data
    train_norm, test_norm, normalizer = prepare_data(
        TRAIN_DATA, TRAIN_LABELS, TEST_DATA
    )

    # Train K-means
    kmeans = KMeansClassifier(n_clusters=4, random_state=42)
    kmeans.fit(train_norm, TRAIN_LABELS, verbose=True)

    # Evaluate
    train_pred = kmeans.predict(train_norm)
    train_acc = np.mean(train_pred == TRAIN_LABELS)

    print(f"\nTraining Accuracy: {train_acc*100:.2f}%")

    # Test predictions
    test_pred = kmeans.predict(test_norm)
    print(f"\nTest Predictions:")
    for i, pred in enumerate(test_pred):
        print(f"  Day {i+1}: Class {pred}")

    print("\n" + "=" * 60)
    print("✓ K-means Classifier test completed!")
    print("=" * 60)
