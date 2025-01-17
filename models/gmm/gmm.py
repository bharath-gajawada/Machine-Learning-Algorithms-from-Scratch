import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        self.memberships = None
        self.centers = None
        self.covariances = None
        self.weights = None
        np.random.seed(42)

    def log_sum_exp(self,arr):
        max_arr = np.max(arr, axis=1, keepdims=True)
        stable_arr = arr - max_arr
        log_sum_exp = np.log(np.sum(np.exp(stable_arr), axis=1, keepdims=True))
        total_log_sum_exp = max_arr + log_sum_exp
        return total_log_sum_exp, stable_arr
    
    def E_step(self, X):
        log_likelihoods = np.array([multivariate_normal(mean=self.centers[c], cov=self.covariances[c], allow_singular=True).logpdf(X)
                                    for c in range(self.n_components)]).T
        log_likelihoods += np.log(self.weights + np.finfo(float).eps)
        
        _, cluster_log_likelihoods = self.log_sum_exp(log_likelihoods)
        total_sample_log_likelihood = np.log(np.sum(np.exp(cluster_log_likelihoods), axis=1, keepdims=True))
        self.memberships = np.exp(cluster_log_likelihoods - total_sample_log_likelihood)

    def M_step(self, X):
        weighted_sums = np.sum(self.memberships, axis=0)
        self.weights = weighted_sums / X.shape[0]
        self.centers = (self.memberships.T @ X) / weighted_sums[:, np.newaxis]

        for c in range(self.n_components):
            diff = X - self.centers[c]
            memberships_c = self.memberships[:, c][:, np.newaxis]
            weighted_diff = memberships_c * diff

            self.covariances[c] = (weighted_diff.T @ diff) / weighted_sums[c]

        self.covariances += np.identity(X.shape[1]) * 1e-6

    def fit(self, X):
        n_samples = X.shape[0]
        self.centers = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.cov(X, rowvar=False)] * self.n_components)
        self.weights = np.full(self.n_components, 1 / self.n_components)

        prev_log_likelihood = -np.inf

        for _ in range(self.max_iter):
            self.E_step(X)
            self.M_step(X)

            log_likelihood = self.get_likelihood(X)
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break

            prev_log_likelihood = log_likelihood

    def get_likelihood(self, X):
        log_likelihoods = np.array([multivariate_normal(mean=self.centers[c], cov=self.covariances[c], allow_singular=True).logpdf(X)
                                    for c in range(self.n_components)]).T
        log_likelihoods += np.log(self.weights + np.finfo(float).eps)
        
        total_sample_log_likelihood, _ = self.log_sum_exp(log_likelihoods)
        return np.sum(total_sample_log_likelihood)/X.shape[0]
    
    def get_params(self):
        return {'centers': self.centers, 'covariances': self.covariances, 'weights': self.weights}
    
    def get_membership(self):
        return self.memberships
    