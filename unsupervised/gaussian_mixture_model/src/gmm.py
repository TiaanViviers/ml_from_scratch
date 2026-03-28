import numpy as np


class GMM():
    def __init__(
        self,
        n_components=1,
        max_iter=100,
        tol=1e-3,
        reg_covar=1e-6,
        random_state=None,
    ):
        if not isinstance(n_components, int):
            raise TypeError(
                f"n_components must be an integer, got {type(n_components).__name__}"
            )
        if n_components <= 0:
            raise ValueError(f"n_components must be > 0, got {n_components}")

        if not isinstance(max_iter, int):
            raise TypeError(f"max_iter must be an integer, got {type(max_iter).__name__}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")

        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol}")
        if reg_covar < 0:
            raise ValueError(f"reg_covar must be >= 0, got {reg_covar}")

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.responsibilities_ = None
        self.labels_ = None
        self.log_likelihood_ = None
        self.log_likelihood_history_ = []
        self.n_iter_ = 0
        self.converged_ = False

    ############################################################################
    # Public API
    ############################################################################
    def fit(self, X):
        X = self._validate_input(X)

        self._initialize_parameters(X)
        self.log_likelihood_history_ = []
        self.log_likelihood_ = None
        self.labels_ = None
        self.n_iter_ = 0
        self.converged_ = False

        previous_log_likelihood = None

        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_ = log_likelihood
            self.log_likelihood_history_.append(log_likelihood)
            self.n_iter_ = iteration + 1

            if previous_log_likelihood is not None:
                if abs(log_likelihood - previous_log_likelihood) <= self.tol:
                    self.converged_ = True
                    break

            previous_log_likelihood = log_likelihood

        self.responsibilities_ = self._e_step(X)
        self.labels_ = np.argmax(self.responsibilities_, axis=0)

        return self

    def predict_proba(self, X):
        X = self._validate_input(X)

        if self.means_ is None or self.covariances_ is None or self.weights_ is None:
            raise RuntimeError("Call fit before predict_proba.")
        if X.shape[0] != self.means_.shape[0]:
            raise ValueError(
                "Feature mismatch between X and fitted means: "
                f"X has {X.shape[0]} features, means have {self.means_.shape[0]}."
            )

        responsibilities = self._e_step(X)
        return responsibilities

    def predict(self, X):
        responsibilities = self.predict_proba(X)
        labels = np.argmax(responsibilities, axis=0)
        return labels

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    ############################################################################
    # Helper methods
    ############################################################################
    
    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D with shape (d, N), got {X.ndim}D.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"X must be non-empty, got shape {X.shape}.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values (NaN or inf).")
        if self.n_components > X.shape[1]:
            raise ValueError(
                "n_components cannot exceed the number of observations: "
                f"got n_components={self.n_components} and N={X.shape[1]}."
            )

        return X


    def _initialize_parameters(self, X):
        rng = np.random.default_rng(self.random_state)
        n_features, n_samples = X.shape

        indices = rng.choice(n_samples, size=self.n_components, replace=False)
        self.means_ = X[:, indices].copy()

        empirical_covariance = np.cov(X, bias=True)
        if empirical_covariance.ndim == 0:
            empirical_covariance = np.array([[empirical_covariance]])

        empirical_covariance = empirical_covariance + self.reg_covar * np.eye(n_features)
        self.covariances_ = np.repeat(
            empirical_covariance[np.newaxis, :, :],
            self.n_components,
            axis=0
        )

        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        self.responsibilities_ = np.full(
            (self.n_components, n_samples),
            1.0 / self.n_components
        )


    def _estimate_log_gaussian_prob(self, X):
        n_features, n_samples = X.shape
        log_prob = np.zeros((self.n_components, n_samples))

        for component_index in range(self.n_components):
            mean = self.means_[:, component_index]
            covariance = self.covariances_[component_index]

            sign, log_det = np.linalg.slogdet(covariance)
            if sign <= 0:
                raise ValueError("Covariance matrix must be positive definite.")

            diff = X - mean[:, np.newaxis]
            solved = np.linalg.solve(covariance, diff)
            mahalanobis = np.sum(diff * solved, axis=0)

            log_prob[component_index, :] = -0.5 * (
                n_features * np.log(2.0 * np.pi) + log_det + mahalanobis
            )

        return log_prob


    def _logsumexp(self, A, axis=0):
        max_values = np.max(A, axis=axis, keepdims=True)
        stabilized = A - max_values
        sum_exp = np.sum(np.exp(stabilized), axis=axis, keepdims=True)
        log_sum_exp = max_values + np.log(sum_exp)
        return log_sum_exp


    def _e_step(self, X):
        log_gaussian_prob = self._estimate_log_gaussian_prob(X)
        log_weights = np.log(self.weights_)[:, np.newaxis]
        weighted_log_prob = log_weights + log_gaussian_prob

        log_prob_norm = self._logsumexp(weighted_log_prob, axis=0)
        log_responsibilities = weighted_log_prob - log_prob_norm
        responsibilities = np.exp(log_responsibilities)

        return responsibilities


    def _m_step(self, X, responsibilities):
        n_features, n_samples = X.shape
        component_weights = np.sum(responsibilities, axis=1)
        rng = np.random.default_rng(self.random_state)

        # Prevent a component from collapsing numerically to exactly zero weight.
        component_weights = np.maximum(component_weights, 1e-12)

        self.weights_ = component_weights / np.sum(component_weights)
        self.means_ = (X @ responsibilities.T) / component_weights[np.newaxis, :]

        covariances = np.zeros((self.n_components, n_features, n_features))

        for component_index in range(self.n_components):
            if component_weights[component_index] <= 1e-10:
                random_index = rng.choice(n_samples)
                self.means_[:, component_index] = X[:, random_index]
                covariances[component_index] = self.reg_covar * np.eye(n_features)
                continue

            diff = X - self.means_[:, component_index][:, np.newaxis]
            weighted_diff = diff * responsibilities[component_index, :][np.newaxis, :]
            covariance = weighted_diff @ diff.T
            covariance = covariance / component_weights[component_index]
            covariance = covariance + self.reg_covar * np.eye(n_features)
            covariances[component_index] = covariance

        self.covariances_ = covariances


    def _compute_log_likelihood(self, X):
        log_gaussian_prob = self._estimate_log_gaussian_prob(X)
        log_weights = np.log(self.weights_)[:, np.newaxis]
        weighted_log_prob = log_weights + log_gaussian_prob
        log_likelihood = np.sum(self._logsumexp(weighted_log_prob, axis=0))
        return log_likelihood
