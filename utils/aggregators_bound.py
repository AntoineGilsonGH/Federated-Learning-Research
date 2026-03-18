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
        
        original_device = vectors.device
        vectors = vectors.to(torch.device(CURRENT_DEVICE))
        
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
            all_indices = torch.arange(n_clients, device=vectors.device)
            mask = torch.ones(n_clients, dtype=torch.bool, device=vectors.device)
            mask[indices_to_keep] = False
            self.last_removed_indices = all_indices[mask].cpu().tolist()
        else:
            indices_to_keep = torch.arange(n_clients)
            self.last_removed_indices = []

        # Return mean of selected vectors
        selected_vectors = vectors[indices_to_keep]
        return torch.mean(selected_vectors, dim=0).to(original_device)

    def _power_method(self, centered, n_iterations=20):
        """
        Power iteration to find principal eigenvector.

        Key trick: with n_clients << dim (e.g. 100 << 500k), we work in
        client space (n x n) instead of gradient space (n x d).
        The top eigenvector of X^T X equals X^T u / ||X^T u|| where u is
        the top eigenvector of X X^T — which is only (n_clients x n_clients).
        This reduces cost from O(n * d) per iteration to O(n^2) for eigen +
        one O(n * d) projection at the end.
        """
        centered = centered.to(torch.device(CURRENT_DEVICE))
        n_clients, dim = centered.shape

        # Gram matrix: (n_clients x n_clients) — cheap when n_clients=100
        gram = centered @ centered.T   # (n, n)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        u = eigenvectors[:, -1]        # top eigenvector in client space

        # Map back to gradient space
        v = centered.T @ u             # (d,)
        v = v / torch.norm(v)
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
        
        original_device = vectors.device
        vectors = vectors.to(torch.device(CURRENT_DEVICE))
        
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
            all_indices = torch.arange(n_clients, device=vectors.device)
            mask = torch.ones(n_clients, dtype=torch.bool, device=vectors.device)
            mask[indices_to_keep] = False
            self.last_removed_indices = all_indices[mask].cpu().tolist()
        else:
            indices_to_keep = torch.arange(n_clients)
            self.last_removed_indices = []

        selected_vectors = vectors[indices_to_keep]
        return torch.mean(selected_vectors, dim=0).to(original_device)

    def _randomized_svd(self, X, n_components, n_iterations=5):
        """
        Efficient SVD exploiting n_clients << dim.

        When n_clients (e.g. 100) << n_features (e.g. 500k), the rank of X
        is at most n_clients, so we compute the full SVD of the small
        (n_clients x n_clients) Gram matrix instead of iterating on (n x d).
        Cost: O(n^2 * d) for the Gram + O(n^3) for eigen — much cheaper.
        """
        X = X.to(torch.device(CURRENT_DEVICE))
        n_samples, n_features = X.shape
        n_components = min(n_components, n_samples)

        # Gram matrix in client space — (n_samples x n_samples)
        gram = X @ X.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)

        # Top n_components eigenvectors (eigh returns ascending order)
        idx = torch.argsort(eigenvalues, descending=True)[:n_components]
        top_eigenvalues = eigenvalues[idx]          # (n_components,)
        top_eigenvectors = eigenvectors[:, idx]     # (n_samples, n_components)

        # Map to feature space: V = X^T U / sqrt(lambda)
        singular_values = torch.sqrt(torch.clamp(top_eigenvalues, min=1e-10))
        V = X.T @ top_eigenvectors / singular_values.unsqueeze(0)  # (d, n_components)

        return V, singular_values


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
        
        original_device = vectors.device
        vectors = vectors.to(torch.device(CURRENT_DEVICE))
        
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
        """
        Power iteration using client-space Gram matrix.
        O(n^2 * d) instead of O(n_iterations * n * d).
        """
        X = X.to(torch.device(CURRENT_DEVICE))
        gram = X @ X.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        u = eigenvectors[:, -1]
        v = X.T @ u
        v = v / torch.norm(v)
        return v


# Registry of available custom aggregators
CUSTOM_AGGREGATORS = {
    "PCAEigenvalueAggregator": PCAEigenvalueAggregator,
    "PCAEigenvalueAggregatorV2": PCAEigenvalueAggregatorV2,
    "RobustPCAEigenvalueAggregator": RobustPCAEigenvalueAggregator,
}