import numpy as np


class LogisticRegression():
    def __init__(self, lambda_reg=1.0, max_iter=100, tol=1e-6, reg_bias=True):
        if lambda_reg <= 0:
            raise ValueError(f"lambda_reg is required to be >0, got {lambda_reg}")
        self.lambda_reg = lambda_reg 
        self.max_iter = max_iter
        self.tol = tol
        self.reg_bias = reg_bias
        
        self.class_labels = None
        self.w = None
        self.n_iter = None
        self.converged = None
    
    
    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        
        # Change label mapping to 0, 1
        y = np.array([0 if i == self.class_labels[0] else 1 for i in y])
        
        # add bias/leading 1s to X
        ones = np.ones(X.shape[1])
        Xb = np.vstack([ones, X])
        
        # initialise weights
        self.w = np.zeros(Xb.shape[0])
        self.n_iter = 0
        self.converged = False
        
        # find optimal weights by Newton's method
        for i in range(self.max_iter):
            self.n_iter += 1
            delta = self._newton_step(Xb, y, self.w)
            self.w = self.w - delta
            if np.linalg.norm(delta) < self.tol:
                self.converged = True
                break
            
        if not self.converged:
            print("WARNING: LogisticRegression did not converge to a solution")
            print("Adjust max_iter or tol (tolerance)")
            
        return self
        
    
    
    def predict_log_proba(self, X_new):
        X_new = np.asarray(X_new, dtype=float)

        if self.class_labels is None or self.w is None:
            raise RuntimeError("Call fit before predict_log_proba.")
        if X_new.ndim != 2:
            raise ValueError(f"X must be 2D, got {X_new.ndim}D.")
        if X_new.shape[0] != len(self.w) - 1:
            raise ValueError(
                f"Feature mismatch: X has {X_new.shape[0]} features, expected {len(self.w)-1}."
            )
            
        # add bias/leading 1s to X
        ones = np.ones(X_new.shape[1])
        Xb_new = np.vstack([ones, X_new])
        
        # compute probabilities
        p1 = self._compute_probabilities(Xb_new, self.w)
        p0 = 1 - p1
        
        #clip 0 probabilities before logs
        log_p1 = np.log(np.clip(p1, 1e-12, 1-1e-12))
        log_p0 = np.log(np.clip(p0, 1e-12, 1-1e-12))
        
        return np.vstack((log_p0, log_p1))
        
    
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
    



    def _validate_input(self, X, y):
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
        self.class_labels = np.unique(y)
        if len(self.class_labels) != 2:
            raise ValueError("LogisticRegression only supports 2 distinct classes.")

        return X, y
    
    
    def _sigmoid(self, a):
        a = np.asarray(a, dtype=float)
        s = np.empty_like(a, dtype=float)

        pos_mask = a >= 0
        neg_mask = ~pos_mask

        s[pos_mask] = 1.0 / (1.0 + np.exp(-a[pos_mask]))
        exp_a = np.exp(a[neg_mask])
        s[neg_mask] = exp_a / (1.0 + exp_a)

        return s
    
    
    def _compute_probabilities(self, Xb, w):
        a = w.T @ Xb
        p = self._sigmoid(a)
        return p


    def _compute_gradient(self, Xb, y, w):
        p = self._compute_probabilities(Xb, w)
        r = p - y

        g_data = Xb @ r

        g_reg = (1.0 / self.lambda_reg) * w
        if not self.reg_bias:
            g_reg = g_reg.copy()
            g_reg[0] = 0.0

        g = g_data + g_reg
        return g


    def _compute_hessian(self, Xb, w):
        p = self._compute_probabilities(Xb, w)
        
        # compute curvature weights
        r = p * (1 - p)
        
        # build data Hessian
        Xw = Xb * r
        H_data = Xw @ Xb.T
        
        # regularisation Hessian
        H_reg = (1/self.lambda_reg) * np.eye(Xb.shape[0])
        if self.reg_bias == False:
            H_reg[0][0] = 0
            
        return H_data + H_reg
    
    
    def _newton_step(self, Xb, y, w):
        g = self._compute_gradient(Xb, y, w)
        H = self._compute_hessian(Xb, w)
        
        # H*delta = g -> Ax=b
        delta = np.linalg.solve(H, g)
        return delta
    
    