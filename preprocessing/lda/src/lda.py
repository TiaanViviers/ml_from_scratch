import numpy as np

class LDA:
    """Linear Discriminant Analysis for dimensionality reduction.

    This implementation follows the convention used in CS315:
        features are rows and observations are columns.
    """

    def __init__(self, n_components=None, normalise=False):
        """Initialize the LDA model.

        Parameters:
        -----------
        n_components : int | None
            Target dimensionality. If None, the maximum valid dimensionality is used.

        Returns:
        --------
        None
        """
        self.n_components = n_components
        self.normalise = normalise

        self.class_labels = None
        self.global_mean = None
        self.class_means = None
        self.transformation_matrix = None
        self.explained_variance_ratio = None
        

    def fit(self, X, y):
        """Fit LDA projection directions from training data.

        Parameters:
        -----------
        X : np.ndarray
            Training matrix of shape ``(n_features, n_samples)``.
        y : np.ndarray
            Class-label vector of shape ``(n_samples,)``.

        Returns:
        --------
        LDA
            The fitted model instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Validate model parameters and input arrays.
        self._validate_input(X, y)

        # Compute means.
        self.global_mean = np.mean(X, axis=1)
        self.class_means = self._compute_class_means(X, y)

        # Compute scatter matrices.
        S_w = self._compute_within_scatter(X, y)
        class_counts = self._compute_class_counts(X, y)
        S_b = self._compute_between_scatter(X.shape[0], class_counts)
        
        if self.normalise:
            S_w = S_w / X.shape[1]
            S_b = S_b / X.shape[1]

        # Compute whitening matrix.
        P = self._compute_whitening(S_w)
        # Whiten S_b.
        S_b_white = P @ S_b @ P.T
        # Compute directions of maximal between-class variance.
        V_k = self._eigendecompose_S_b_white(S_b_white)
        # Compute transformation matrix.
        self.transformation_matrix = P.T @ V_k
        return self
    

    def transform(self, X_new):
        """Project new data into the fitted LDA subspace.

        Parameters:
        -----------
        X_new : np.ndarray
            Matrix of shape ``(n_features, n_new_samples)``.

        Returns:
        --------
        np.ndarray
            Projected matrix of shape ``(n_components, n_new_samples)``.
        """
        X_new = np.asarray(X_new)

        # Validate that LDA has been fitted.
        if self.global_mean is None or self.transformation_matrix is None:
            raise RuntimeError("Call fit before transform.")
        if np.ndim(X_new) != 2:
            raise ValueError(f"LDA expects X_new to be 2D, got {np.ndim(X_new)}D.")
        if X_new.shape[0] != self.global_mean.shape[0]:
            raise ValueError(
                "X_new has incompatible feature dimension: "
                f"expected {self.global_mean.shape[0]}, got {X_new.shape[0]}."
            )

        # Center new data using training global mean.
        D_new = X_new - np.outer(self.global_mean, np.ones(shape=X_new.shape[1]))

        # Project into LDA space.
        Z = self.transformation_matrix.T @ D_new
        return Z


    def fit_transform(self, X, y):
        """Fit the model and return the projected training data.

        Parameters:
        -----------
        X : np.ndarray
            Training matrix of shape ``(n_features, n_samples)``.
        y : np.ndarray
            Class-label vector of shape ``(n_samples,)``.

        Returns:
        --------
        np.ndarray
            Projected training matrix of shape ``(n_components, n_samples)``.
        """
        self.fit(X, y)
        return self.transform(X)
    

    def _validate_input(self, X, y):
        """Validate input shapes and model hyperparameters.

        This function also sets class labels for `n_components`
        when the value is not explicitly provided.

        Parameters:
        -----------
        X : np.ndarray
            Training matrix of shape ``(n_features, n_samples)``.
        y : np.ndarray
            Class-label vector of shape ``(n_samples,)``.

        Returns:
        --------
        None
        """
        if np.ndim(X) != 2:
            raise ValueError(f"LDA expects X to be 2D, got {np.ndim(X)}D.")
        if np.ndim(y) != 1:
            raise ValueError(f"LDA expects y to be 1D, got {np.ndim(y)}D.")
        if X.shape[1] != y.shape[0]:
            raise ValueError(
                "X and y are misaligned: "
                f"X has {X.shape[1]} samples while y has {y.shape[0]} labels."
            )

        self.class_labels = np.unique(y)
        if len(self.class_labels) < 2:
            raise ValueError("LDA requires at least 2 distinct classes.")

        if self.n_components is not None:
            if not isinstance(self.n_components, (int, np.integer)):
                raise TypeError("n_components must be an integer or None.")
            if self.n_components < 1:
                raise ValueError("n_components must be at least 1.")

        max_components = min(X.shape[0], len(self.class_labels) - 1)
        if self.n_components is None:
            self.n_components = max_components
        elif self.n_components > max_components:
            raise ValueError(
                f"LDA can produce at most {max_components} components, "
                f"got {self.n_components}."
            )


    def _compute_class_means(self, X, y):
        """Compute the mean feature vector for each class.

        Parameters:
        -----------
        X : np.ndarray
            Training matrix of shape ``(n_features, n_samples)``.
        y : np.ndarray
            Class-label vector of shape ``(n_samples,)``.

        Returns:
        --------
        np.ndarray
            Class-mean matrix of shape ``(n_features, n_classes)``.
        """
        class_means = []
        for c in self.class_labels:
            mask = (y == c)
            X_c = X[:, mask]
            class_means.append(np.mean(X_c, axis=1))

        return np.column_stack(class_means)


    def _compute_within_scatter(self, X, y):
        """Compute within-class scatter matrix.

        Parameters:
        -----------
        X : np.ndarray
            Training matrix of shape ``(n_features, n_samples)``.
        y : np.ndarray
            Class-label vector of shape ``(n_samples,)``.

        Returns:
        --------
        np.ndarray
            Within-class scatter matrix of shape ``(n_features, n_features)``.
        """
        S_w = np.zeros((X.shape[0], X.shape[0]))
        for i, c in enumerate(self.class_labels):
            mask = (y == c)
            X_c = X[:, mask]
            D_c = X_c - np.outer(self.class_means[:, i], np.ones(X_c.shape[1]).T)
            S_w += D_c @ D_c.T

        return S_w


    def _compute_class_counts(self, X, y):
        """Count the number of samples in each class.

        Parameters:
        -----------
        X : np.ndarray
            Training matrix of shape ``(n_features, n_samples)``.
        y : np.ndarray
            Class-label vector of shape ``(n_samples,)``.

        Returns:
        --------
        list[int]
            Per-class sample counts in the order of `self.class_labels`.
        """
        class_counts = []
        for c in self.class_labels:
            mask = (y == c)
            class_counts.append(int(np.sum(mask)))

        return class_counts


    def _compute_between_scatter(self, d, class_counts):
        """Compute between-class scatter matrix.

        Parameters:
        -----------
        d : int
            Number of original features.
        class_counts : list[int]
            Per-class sample counts in the order of `self.class_labels`.

        Returns:
        --------
        np.ndarray
            Between-class scatter matrix of shape ``(d, d)``.
        """
        S_b = np.zeros((d, d))
        for i in range(len(self.class_labels)):
            m_c = self.class_means[:, i] - self.global_mean
            S_b += class_counts[i] * np.outer(m_c, m_c)

        return S_b


    def _compute_whitening(self, S_w, tol=1e-12):
        """Compute whitening matrix for within-class scatter.

        Some near-zero eigenvalues are removed for numerical stability.

        Parameters:
        -----------
        S_w : np.ndarray
            Within-class scatter matrix.
        tol : float
            Relative tolerance used to drop tiny eigenvalues.

        Returns:
        --------
        np.ndarray
            Whitening matrix ``P`` such that ``P @ S_w @ P.T`` is
            approximately identity on the retained subspace.
        """
        # Eigendecompose S_w.
        eigenvalues, eigenvectors = np.linalg.eigh(S_w)

        # Remove near-zero eigenvalues.
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        max_eig = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
        tol_lvl = tol * max(1.0, max_eig)
        keep = eigenvalues > tol_lvl
        if not np.any(keep):
            raise ValueError(
                "Within-class scatter is numerically singular; no stable "
                "whitening directions were found."
            )

        eigenvalues = eigenvalues[keep]
        eigenvectors = eigenvectors[:, keep]

        # Build whitening matrix.
        A = np.diag(1.0 / np.sqrt(eigenvalues))
        P = A @ eigenvectors.T

        return P


    def _eigendecompose_S_b_white(self, S_b_white):
        """Find dominant discriminative directions in whitened space.

        The explained-variance ratio is stored in `self.explained_variance_ratio`.

        Parameters:
        -----------
        S_b_white : np.ndarray
            Whitened between-class scatter matrix.

        Returns:
        --------
        np.ndarray
            Matrix of top eigenvectors with shape
            ``(retained_dims, n_components)``.
        """
        # Eigendecompose whitened between-class scatter.
        eigenvalues, eigenvectors = np.linalg.eigh(S_b_white)

        # Clip tiny negative values and sort descending.
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Keep top n_components directions.
        eigenvalues_clipped = eigenvalues[:self.n_components]
        V_k = eigenvectors[:, :self.n_components]

        # Compute explained variance ratios safely.
        total_variance = float(np.sum(eigenvalues))
        if total_variance <= 0.0:
            self.explained_variance_ratio = np.zeros(self.n_components)
        else:
            self.explained_variance_ratio = eigenvalues_clipped / total_variance

        return V_k
    
