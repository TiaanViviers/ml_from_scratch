import numpy as np


class KMeans():
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        if not isinstance(n_clusters, int):
            raise TypeError(f"n_clusters must be an integer, got {type(n_clusters).__name__}")
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be > 0, got {n_clusters}")

        if not isinstance(max_iter, int):
            raise TypeError(f"max_iter must be an integer, got {type(max_iter).__name__}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")

        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol}")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labels = None
        self.distortion = None
        self.n_iter = 0
        self.converged = False

    ############################################################################
    # Public API
    ############################################################################
    def fit(self, X):
        X = self._validate_input(X)
        
        self.centroids = self._initialize_centroids(X)
        self.labels = None
        self.distortion = None
        self.n_iter = 0
        self.converged = False

        for iteration in range(self.max_iter):
            self.labels = self._assign_clusters(X, self.centroids)
            updated_centroids = self._update_centroids(X, self.labels)

            centroid_shift = np.linalg.norm(updated_centroids - self.centroids)
            self.centroids = updated_centroids
            self.distortion = self._compute_distortion(X, self.centroids, self.labels)
            self.n_iter = iteration + 1

            if centroid_shift <= self.tol:
                self.converged = True
                break

        return self


    def predict(self, X):
        X = self._validate_input(X)

        if self.centroids is None:
            raise RuntimeError("Call fit before predict.")
        if X.shape[0] != self.centroids.shape[0]:
            raise ValueError(
                "Feature mismatch between X and fitted centroids: "
                f"X has {X.shape[0]} features, centroids have {self.centroids.shape[0]}."
            )

        labels = self._assign_clusters(X, self.centroids)
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
        if self.n_clusters > X.shape[1]:
            raise ValueError(
                "n_clusters cannot exceed the number of observations: "
                f"got n_clusters={self.n_clusters} and N={X.shape[1]}."
            )

        return X


    def _initialize_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        centroid_indices = rng.choice(X.shape[1], size=self.n_clusters, replace=False)
        centroids = X[:, centroid_indices]

        return centroids


    def _assign_clusters(self, X, centroids):
        squared_distances = np.sum(
            (X[:, np.newaxis, :] - centroids[:, :, np.newaxis]) ** 2,
            axis=0
        )
        labels = np.argmin(squared_distances, axis=0)

        return labels


    def _update_centroids(self, X, labels):
        centroids = np.zeros((X.shape[0], self.n_clusters))
        rng = np.random.default_rng(self.random_state)

        for cluster_index in range(self.n_clusters):
            cluster_points = X[:, labels == cluster_index]

            if cluster_points.shape[1] == 0:
                random_index = rng.choice(X.shape[1])
                centroids[:, cluster_index] = X[:, random_index]
            else:
                centroids[:, cluster_index] = np.mean(cluster_points, axis=1)

        return centroids


    def _compute_distortion(self, X, centroids, labels):
        squared_distances = np.sum((X - centroids[:, labels]) ** 2, axis=0)
        distortion = np.sum(squared_distances)

        return distortion
