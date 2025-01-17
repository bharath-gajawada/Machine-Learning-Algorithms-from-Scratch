import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        self.components = Vt[:self.n_components]
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)
    
    def checkPCA(self, X):
        X_transformed = self.transform(X)

        return X_transformed.shape[1] == self.n_components