import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) for matrices with shape (d, N).
    
    This implementation follows the CS315 module convention:
    - rows are features/variables (dimension d)
    - columns are observations/samples (count N)
    
    The class supports two fitting routes:
    - `fit_svd`: PCA via singular value decomposition (SVD) of centred data
    - `fit_eigh`: PCA via eigendecomposition of the covariance matrix
    
    Stored attributes after fitting:
    - `x_bar` : feature-wise training mean vector, shape (d,)
    - `principle_components` : matrix whose columns are retained principal directions, shape (d, k)
    - `explained_variance` : eigenvalues for retained components, shape (k,)
    - `explained_variance_ratio` : retained variance fractions, shape (k,)
    """
    
    def __init__(self, n_components=None, whiten=False, ddof=None):
        """
        Initialise PCA hyperparameters and fitted-state placeholders.
        
        Parameters:
        -----------
        n_components : int or None, default=None
            Number of principal components to retain.
            If None, uses the maximum valid value for the chosen fit method.
        whiten : bool, default=False
            If True, transformed coordinates are scaled by inverse standard
            deviation of each retained component.
        ddof : int or None, default=None
            Delta degrees of freedom used in covariance/eigenvalue scaling.
            If None, defaults to 0.
        """
        self.n_components = n_components
        self.whiten = whiten
        self.ddof = 0 if ddof is None else ddof
        
        # Mean vector of features calculated on 'training' data (used for centering)
        self.x_bar = None
        # Columns are principal directions (eigenvectors of covariance matrix)
        self.principle_components = None
        # Eigenvalues of covariance matrix (variance captured per principal component)
        self.explained_variance = None
        # Fraction of total variance explained by each principal component
        self.explained_variance_ratio = None
        
    #===========================================================================
    # External API
    #===========================================================================

    def fit_svd(self, X):
        """
        Perform SVD to derive the principal components and singular values.
        
        Parameters:
        -----------
        X : ndarray of shape (d, N)
            The input matrix where `d` represents the feature dimension 
            and `N` is the number of observations.
            
        Returns:
        --------
        None
        """
        self._set_n_components(X, method="svd")
        
        # Perform Singular Value Decomposition
        D = self._centre_train(X)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        
        self.principle_components = U[:, :self.n_components]
        singular_values = S[:self.n_components]
        # compute total varaince based on all singular values
        total_variance = np.sum(self._expl_var(S, D.shape[1]))
        self.explained_variance = self._expl_var(singular_values, D.shape[1])
        self._set_expl_var_ratios(total_variance)
        

    def fit_eigh(self, X):
        """
        Perform covariance-based PCA using eigendecomposition.
        
        This method:
        - centres training data
        - builds covariance matrix S = (1/(N-ddof)) * D D^T
        - eigendecomposes S
        - sorts eigenpairs in descending-variance order
        - stores retained principal directions and explained variance data
        
        Parameters:
        -----------
        X : ndarray of shape (d, N)
            The input matrix where `d` represents the feature dimension
            and `N` is the number of observations.
        
        Returns:
        --------
        None
        """
        self._set_n_components(X, method="eigh")
            
        D = self._centre_train(X)
        S = self._build_cov(D)
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        #sort eigenvalues and eigenvectors in non-increasing order
        eigenvalues = np.flip(eigenvalues)
        eigenvectors = np.fliplr(eigenvectors)
        
        self.principle_components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        total_variance = sum(eigenvalues)
        self._set_expl_var_ratios(total_variance)
            
    
    def transform(self, X_new):
        """
        Project data into the fitted principal-component coordinate system.
        
        This method first centres `X_new` with the training mean `x_bar`, then
        projects onto retained principal directions. If whitening is enabled,
        each component score is additionally divided by sqrt(eigenvalue) so
        each retained component has unit variance.
        
        Parameters:
        -----------
        X_new : ndarray of shape (d, M)
            Data to project, where rows are features and columns are
            observations.
        
        Returns:
        --------
        Z : ndarray of shape (k, M)
            Projected coordinates (component scores) in retained PC basis.
        """
        if self.principle_components is None:
            raise RuntimeError("Please fit PCA before calling transform.")
        
        D_new = self._centre_oos(X_new)
        if self.whiten:
            if np.any(self.explained_variance <= 0):
                raise ValueError(
                    "Whitening requires strictly positive explained variance "
                    "for all retained components. Reduce n_components or disable whiten."
                )
            A = np.diag(np.ones(shape=len(self.explained_variance)) / np.sqrt(self.explained_variance))
            Z = A @ self.principle_components.T @ D_new
        else: 
            Z = self.principle_components.T @ D_new
        
        return Z
    
    
    def fit_svd_transform(self, X):
        """
        Convenience wrapper equivalent to:
        1) `fit_svd(X)`
        2) `transform(X)`
        
        Parameters:
        -----------
        X : ndarray of shape (d, N)
            Training data matrix with rows as features and columns as
            observations.
        
        Returns:
        --------
        Z : ndarray of shape (k, N)
            Projected training data in retained PC basis.
        """
        self.fit_svd(X)
        return self.transform(X)
    
    
    def fit_eigh_transform(self, X):
        """
        Convenience wrapper equivalent to:
        1) `fit_eigh(X)`
        2) `transform(X)`
        
        Parameters:
        -----------
        X : ndarray of shape (d, N)
            Training data matrix with rows as features and columns as
            observations.
        
        Returns:
        --------
        Z : ndarray of shape (k, N)
            Projected training data in retained PC basis.
        """
        self.fit_eigh(X)
        return self.transform(X)
    
    
    def inverse_transform(self, Z):
        """
        Map component scores back to the original feature space.
        
        If whitening was used during `transform`, this method first reverses
        the whitening scale and then reconstructs centred data with the
        retained basis before adding back the training mean vector.
        
        Parameters:
        -----------
        Z : ndarray of shape (k, M)
            Component scores in retained PC basis.
        
        Returns:
        --------
        X_hat : ndarray of shape (d, M)
            Reconstructed data in the original feature space.
        """
        if self.principle_components is None:
            raise RuntimeError("Please fit PCA before calling inverse_transform.")
        
        if self.whiten:
            A = np.diag(np.sqrt(self.explained_variance))
            Z = A @ Z
        
        D_hat = self.principle_components @ Z
        x_bar_matrix = np.outer(self.x_bar, np.ones(shape=D_hat.shape[1]).T)
        X_hat = D_hat +  x_bar_matrix
        return X_hat


    #===========================================================================
    # Internal Helpers
    #===========================================================================
    
    def _set_n_components(self, X, method):
        """
        Validate and set the number of principal components.
        
        For SVD with X in shape (d, N), at most min(d, N) singular directions
        can be recovered. For covariance eigendecomposition, the maximum count
        is d.
        
        Parameters:
        -----------
        X : ndarray of shape (d, N)
            Input data matrix using rows=features and columns=observations.
        method : {"svd", "eigh"}
            Fit method being used, which determines the maximum allowable
            number of retained components.
        
        Returns:
        --------
        None
        """
        d, N = X.shape
        max_components = min(d, N) if method == "svd" else d

        if self.n_components is None:
            self.n_components = max_components
            return

        if not isinstance(self.n_components, (int, np.integer)):
            raise TypeError("n_components must be an integer or None")
        if self.n_components <= 0:
            raise ValueError("n_components must be >= 1")
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds maximum "
                f"allowed ({max_components}) for method='{method}' and "
                f"input shape {X.shape}"
            )

    
    def _centre_train(self, X):
        """
        Centre the data by subtracting the mean vector x_bar from each column.
        
        This method also sets the x_bar class variable
        
        Parameters:
        -----------
        X : ndarray of shape (d, N)
            The input matrix where `d` represents the feature dimension 
            and `N` is the number of observations.
            
        Returns:
        --------
        D : ndarray of shape (d, N)
            The Centred data matrix of the "training" observations
        """
        self.x_bar = np.mean(X, axis=1)
        x_bar_matrix = np.outer(self.x_bar, np.ones(shape=X.shape[1]).T)
        D = X - x_bar_matrix
        return D
    
    
    def _centre_oos(self, X_new):
        """
        Centre out-of-sample data using the stored training mean vector.
        
        Parameters:
        -----------
        X_new : ndarray of shape (d, M)
            New data matrix where rows are features and columns are
            observations.
        
        Returns:
        --------
        D_new : ndarray of shape (d, M)
            Centred out-of-sample data matrix.
        """
        x_bar_matrix = np.outer(self.x_bar, np.ones(shape=X_new.shape[1]).T)
        D_new = X_new - x_bar_matrix
        return D_new


    def _build_cov(self, D):
        """
        Build covariance matrix from centred data.
        
        Covariance is computed as:
            S = (1/(N-ddof)) * D D^T
        where D has shape (d, N), giving S shape (d, d).
        
        Parameters:
        -----------
        D : ndarray of shape (d, N)
            Centred data matrix.
        
        Returns:
        --------
        S : ndarray of shape (d, d)
            Covariance matrix.
        """
        if D.shape[1] - self.ddof <= 0:
            raise ValueError(
                f"Invalid ddof={self.ddof} for N={D.shape[1]}. "
                "Require N - ddof > 0."
            )
        S = (1 / (D.shape[1] - self.ddof)) * (D @ D.T)
        return S
    
    
    def _expl_var(self, sing_vals, N):
        """
        Calculate explained variance as squared singular values divided by (N-ddof).
        Singular values are related to eigenvalues via:
            lambda_i = (sing_vals)_i^2 / (N - ddof)
        
        Parameters:
        -----------
        sing_vals : np.ndarray
            Singular values corresponding to centred data.
        N : int
            The number of observations in the "training" set
        
        Returns:
        --------
        explained_variance : ndarray of shape (len(sing_vals),)
            The explained variance (eigenvalues) based on the provided singular values.
        """
        if N - self.ddof <= 0:
            raise ValueError(
                f"Invalid ddof={self.ddof} for N={N}. Require N - ddof > 0."
            )
        explained_variance = np.empty(len(sing_vals))
        for i, s in enumerate(sing_vals):
            explained_variance[i] = s**2 / (N - self.ddof)
        
        return explained_variance
        
        
    def _set_expl_var_ratios(self, total_variance):
        """
        Set explained variance ratios as:
            explained_variance[i] / sum(explained_variance)
        
        Parameters:
        -----------
        total_variance : float
            Total variance in the full (non-truncated) eigenspectrum used as
            denominator for ratio computation.
        
        Returns:
        --------
        None, just sets the class variable 'explained_variance_ratio'
        """
        if np.ndim(total_variance) != 0:
            raise ValueError("total_variance must be a scalar.")
        if total_variance <= 0:
            raise ValueError("total_variance must be strictly positive.")

        self.explained_variance_ratio = np.empty(len(self.explained_variance))
        
        for i, lambda_ in enumerate(self.explained_variance):
            self.explained_variance_ratio[i] = lambda_ / total_variance
