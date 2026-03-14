import numpy as np

class GaussianNB:
    def __init__(self, ):
        
        self.class_labels = None
        self.class_counts = None
        self.priors = None
        self.means = None
        self.variances = None
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   External API
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def fit(self, X, y):
        # Coerce inputs to NumPy arrays for consistent indexing.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        # Validate core shapes and values
        if X.ndim != 2:
            raise ValueError(f"X must be 2D with shape (d, N), got {X.ndim}D.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D with shape (N,), got {y.ndim}D.")
        if X.shape[1] != y.shape[0]:
            raise ValueError(
                "X and y are misaligned: "
                f"X has {X.shape[1]} observations, y has {y.shape[0]} labels."
            )
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"X must be non-empty, got shape {X.shape}.")
        if y.shape[0] == 0:
            raise ValueError("y must be non-empty.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values (NaN or inf).")
        
        # compute class labels and counts
        self.class_labels, self.class_counts = np.unique(y, return_counts=True)
        if len(self.class_labels) < 2:
            raise ValueError("GaussianNB requires at least 2 distinct classes.")

        # compute priors
        self.priors = self._compute_priors(X)
        # compute mean feature vector for each class
        self.means = self._compute_means(X, y)
        # compute varaince feature vector for each class
        self.variances = self._compute_variances(X, y)
        
    
    def predict_log_proba(self, X_new):
        X_new = np.asarray(X_new, dtype=float)

        if self.class_labels is None or self.means is None or self.variances is None or self.priors is None:
            raise RuntimeError("Call fit before predict_log_proba.")
        if X_new.ndim != 2:
            raise ValueError(f"X must be 2D, got {X_new.ndim}D.")
        if X_new.shape[0] != self.means.shape[0]:
            raise ValueError(
                f"Feature mismatch: X has {X_new.shape[0]} features, expected {self.means.shape[0]}."
            )
        
        scores = [] #k x N
        for k in range(len(self.class_labels)):
            mu_k = self.means[:, k]
            sigma2_k = self.variances[:, k]
            prior_k = self.priors[k]
            
            score_k = self._compute_score(X_new, prior_k, mu_k, sigma2_k)
            scores.append(score_k)
        scores = np.array(scores)
            
        log_probas = self._normalise_scores(scores)
        return log_probas
    
    
    def predict_proba(self, X_new):
        log_probas = self.predict_log_proba(X_new)
        probas = np.exp(log_probas)
        return probas
    
    
    def predict(self, X_new, as_numeric=False):
        probas = self.predict_proba(X_new)
        pred_indices = np.argmax(probas, axis=0)
        
        if as_numeric:
            return pred_indices
        
        preds = []
        for index in pred_indices:
            preds.append(self.class_labels[index])
        return np.array(preds)
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   Internal Helpers
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def _compute_priors(self, X):
        n = X.shape[1]
        priors = np.empty(len(self.class_labels))
    
        for i, count in enumerate(self.class_counts):
            priors[i] = count / n
        
        return priors
    
    
    def _compute_means(self, X, y):
        means = []
        
        for label in self.class_labels:
            mask = (y==label)
            X_c = X[:, mask]
            means.append(np.mean(X_c, axis=1))   
            
        return np.column_stack(means)        
            

    def _compute_variances(self, X, y):
        variances = []
        
        for label in self.class_labels:
            mask = (y==label)
            X_c = X[:, mask]
            variances.append(np.var(X_c, axis=1, ddof=0))
            
        variances = np.maximum(np.column_stack(variances), 1e-12)
        return variances
        
        
    def _compute_score(self, X, prior, mus, sigma2s):
        score = []
        
        for j in range(X.shape[1]):
            sample_score = np.log(prior)
            for n in range(len(mus)):
                sample_score += self._univ_log_gaussian(X[n][j], mus[n], sigma2s[n])
            score.append(sample_score)
                
        return np.array(score)
    
        
    def _univ_log_gaussian(self, x, mu, sigma2):
        return -0.5 * np.log(2*np.pi*sigma2) - 0.5 * ((x - mu)**2 / sigma2)
        
        
    def _normalise_scores(self, scores):
        total_cols = scores.shape[1]
        log_probas = np.empty((scores.shape[0], scores.shape[1]))  

        for i in range(total_cols):
            col = scores[:, i]
            col_max = np.max(col)
            shifted_col = col - col_max
            e_col = np.exp(shifted_col)
            S = np.sum(e_col)
            log_denom = col_max + np.log(S)
            log_probas[:, i] = col - log_denom
            
        return log_probas
        
    
    
