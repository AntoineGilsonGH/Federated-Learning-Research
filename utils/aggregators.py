"""
Memory-efficient custom aggregators for ByzFL framework.
These avoid creating large covariance matrices.
"""

import torch
import numpy as np

from config import DEVICE as CURRENT_DEVICE

# CURRENT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PCAEigenvalueAggregator:
    """
    Memory-efficient PCA-based Eigenvalue Aggregator (PCAEA).
    Uses randomized SVD and avoids creating full covariance matrix.
    """

    def __init__(self, f=0, n_components=10, use_subsampling=True, subsample_ratio=0.1, **kwargs):
        """
        Args:
            f: Number of Byzantine clients to tolerate
            n_components: Number of principal components to use (default: 10)
            use_subsampling: Whether to subsample dimensions for covariance computation
            subsample_ratio: Ratio of dimensions to subsample (if use_subsampling=True)
        """
        self.f = f
        self.n_components = n_components
        self.use_subsampling = use_subsampling
        self.subsample_ratio = subsample_ratio

    def __call__(self, vectors):
        # Handle list input
        if isinstance(vectors, list):
            if len(vectors) == 0:
                return torch.tensor([])
            vectors = torch.stack(vectors)
        
        # Ensure on CPU for large computations to avoid GPU OOM
        original_device = vectors.device
        vectors = vectors  # .cpu()
        
        n_clients = vectors.shape[0]
        dim = vectors.shape[1]
        
        # If no clients or f >= n_clients, return mean of all
        if n_clients == 0 or self.f >= n_clients:
            return torch.mean(vectors, dim=0).to(original_device)

        # Step 1: Compute mean
        mu = torch.mean(vectors, dim=0)
        centered = vectors - mu

        # Step 2: Use randomized PCA to find principal components efficiently
        # This avoids computing the full covariance matrix
        
        # Option A: Subsampling for very high dimensions
        if self.use_subsampling and dim > 10000:
            # Randomly subsample dimensions
            subsample_size = min(int(dim * self.subsample_ratio), 5000)
            indices = torch.randperm(dim)[:subsample_size]
            centered_subsampled = centered[:, indices]
            
            # Compute covariance on subsampled data (much smaller)
            cov_subsampled = torch.mm(centered_subsampled.T, centered_subsampled) / n_clients
            
            # Get principal eigenvector on subsampled space
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_subsampled)
            principal_dir_subsampled = eigenvectors[:, -1]
            
            # Project all original vectors using the subsampled direction
            # (this is an approximation)
            projections = torch.abs(centered_subsampled @ principal_dir_subsampled)
        
        # Option B: Use randomized SVD for better accuracy
        else:
            # Use power iteration to find principal component
            # This avoids creating the full covariance matrix
            principal_dir = self._power_method(centered, n_iterations=20)
            projections = torch.abs(centered @ principal_dir)

        # Keep indices with smallest scores
        if self.f > 0:
            _, indices_to_keep = torch.topk(projections, n_clients - self.f, largest=False)
        else:
            indices_to_keep = torch.arange(n_clients)

        # Return mean of selected vectors
        selected_vectors = vectors[indices_to_keep]
        return torch.mean(selected_vectors, dim=0).to(original_device)

    def _power_method(self, centered, n_iterations=20):
        """Power iteration to find principal eigenvector without covariance matrix."""
        n_clients, dim = centered.shape
        
        centered = centered.to(torch.device(CURRENT_DEVICE))
        
        # Initialize random vector
        v = torch.randn(dim).to(torch.device(CURRENT_DEVICE))
        v = v / torch.norm(v)
        
        for _ in range(n_iterations):
            # Compute (X^T X) v efficiently: X^T (X v)
            # This avoids creating the full covariance matrix
            Xv = centered @ v  # shape: (n_clients,)
            Xt_Xv = centered.T @ Xv  # shape: (dim,)
            
            # Normalize
            v_new = Xt_Xv / torch.norm(Xt_Xv)
            
            # Check convergence
            if torch.allclose(v, v_new, rtol=1e-6):
                break
            v = v_new
        
        return v


class PCAEigenvalueAggregatorV2:
    """
    Memory-efficient version using multiple components via randomized SVD.
    """

    def __init__(self, f=0, n_components=5, variance_threshold=None, **kwargs):
        """
        Args:
            f: Number of Byzantine clients to tolerate
            n_components: Number of components to use (default: 5)
            variance_threshold: If set, adapt n_components to explain this variance
        """
        self.f = f
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def __call__(self, vectors):
        # Handle list input
        if isinstance(vectors, list):
            if len(vectors) == 0:
                return torch.tensor([])
            vectors = torch.stack(vectors)
        
        # Move to CPU for safety
        original_device = vectors.device
        vectors = vectors  # .cpu()
        
        n_clients = vectors.shape[0]
        
        if n_clients == 0 or self.f >= n_clients:
            return torch.mean(vectors, dim=0).to(original_device)

        # Compute mean and center
        mu = torch.mean(vectors, dim=0)
        centered = vectors - mu

        # Use randomized SVD to get top components efficiently
        components, explained_variance = self._randomized_svd(centered, self.n_components)

        # Determine number of components to use based on variance threshold
        if self.variance_threshold is not None:
            explained_variance_ratio = explained_variance / explained_variance.sum()
            cumulative_ratio = torch.cumsum(explained_variance_ratio, dim=0)
            n_use = torch.searchsorted(cumulative_ratio, self.variance_threshold) + 1
            components = components[:, :n_use]

        # Project onto components
        projections = torch.abs(centered @ components)
        
        # Combine scores (use mean of projections across components)
        combined_scores = torch.mean(projections, dim=1)

        # Keep vectors with smallest scores
        if self.f > 0:
            _, indices_to_keep = torch.topk(combined_scores, n_clients - self.f, largest=False)
        else:
            indices_to_keep = torch.arange(n_clients)

        selected_vectors = vectors[indices_to_keep]
        return torch.mean(selected_vectors, dim=0).to(original_device)

    def _randomized_svd(self, X, n_components, n_iterations=5):
        """
        Efficient randomized SVD without computing full covariance.
        Returns top components and singular values.
        """
        n_samples, n_features = X.shape
        X = X.to(torch.device(CURRENT_DEVICE))
        
        # Generate random matrix
        Q = torch.randn(n_features, n_components).to(torch.device(CURRENT_DEVICE))
        
        # Power iteration for better accuracy
        for _ in range(n_iterations):
            # Y = X @ Q
            Y = X @ Q
            # Q_new = X.T @ Y
            Q_new = X.T @ Y
            # Orthonormalize Q
            Q, _ = torch.linalg.qr(Q_new)
        
        # Project and compute SVD on smaller matrix
        Y = X @ Q
        U_small, S, Vt_small = torch.linalg.svd(Y, full_matrices=False)
        V = Q @ Vt_small.T
        
        return V[:, :n_components], S[:n_components]


class RobustPCAEigenvalueAggregator:
    """
    Memory-efficient robust PCA using iterative filtering with power iteration.
    """

    def __init__(self, f=0, max_iterations=5, removal_ratio=0.1, **kwargs):
        """
        Args:
            f: Number of Byzantine clients to tolerate
            max_iterations: Maximum number of filtering iterations
            removal_ratio: Ratio of vectors to remove in each iteration
        """
        self.f = f
        self.max_iterations = max_iterations
        self.removal_ratio = removal_ratio

    def __call__(self, vectors):
        # Handle list input
        if isinstance(vectors, list):
            if len(vectors) == 0:
                return torch.tensor([])
            vectors = torch.stack(vectors)
        
        # Move to CPU
        original_device = vectors.device
        vectors = vectors  # .cpu()
        
        current_vectors = vectors.clone()
        n_clients = vectors.shape[0]
        total_to_remove = self.f

        if n_clients == 0 or total_to_remove >= n_clients:
            return torch.mean(vectors, dim=0).to(original_device)

        for iteration in range(self.max_iterations):
            if len(current_vectors) <= n_clients - total_to_remove:
                break

            # Compute statistics on current set
            mu = torch.mean(current_vectors, dim=0)
            centered = current_vectors - mu

            # Use power iteration to find principal component efficiently
            principal_dir = self._power_method(centered, n_iterations=15)
            
            # Compute scores
            scores = torch.abs(centered @ principal_dir)

            # Remove worst vectors
            n_remove = max(1, int(len(current_vectors) * self.removal_ratio))
            n_remove = min(n_remove, len(current_vectors) - (n_clients - total_to_remove))
            
            if n_remove > 0:
                _, indices_to_remove = torch.topk(scores, n_remove, largest=True)
                keep_mask = torch.ones(len(current_vectors), dtype=torch.bool)
                keep_mask[indices_to_remove] = False
                current_vectors = current_vectors[keep_mask]

        # Final selection and averaging
        if len(current_vectors) > n_clients - total_to_remove:
            mu = torch.mean(current_vectors, dim=0)
            centered = current_vectors - mu
            principal_dir = self._power_method(centered, n_iterations=15)
            scores = torch.abs(centered @ principal_dir)
            
            n_to_keep = min(n_clients - total_to_remove, len(current_vectors))
            _, indices_to_keep = torch.topk(scores, n_to_keep, largest=False)
            current_vectors = current_vectors[indices_to_keep]

        return torch.mean(current_vectors, dim=0).to(original_device)

    def _power_method(self, X, n_iterations=20):
        """Power iteration to find principal eigenvector."""
        n_samples, n_features = X.shape
        X = X.to(torch.device(CURRENT_DEVICE))

        # Initialize random vector
        v = torch.randn(n_features).to(torch.device(CURRENT_DEVICE))
        v = v / torch.norm(v)
        
        for _ in range(n_iterations):
            # X^T (X v) without forming covariance matrix
            Xv = X @ v
            Xt_Xv = X.T @ Xv
            v_new = Xt_Xv / torch.norm(Xt_Xv)
            
            if torch.allclose(v, v_new, rtol=1e-6):
                break
            v = v_new
        
        return v


# Registry of available custom aggregators
CUSTOM_AGGREGATORS = {
    "PCAEigenvalueAggregator": PCAEigenvalueAggregator,
    "PCAEigenvalueAggregatorV2": PCAEigenvalueAggregatorV2,
    "RobustPCAEigenvalueAggregator": RobustPCAEigenvalueAggregator,
}