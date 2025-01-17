import numpy as np

class KMeans:
    def __init__(self, k, tol=1e-4):
        self.k = k
        self.tol = tol
        self.centroids = None
        np.random.seed(42)

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        while True:
            labels = self.predict(X)
            
            centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            if np.linalg.norm(self.centroids - centroids) < self.tol:
                break
            
            self.centroids = centroids
    
    def predict(self, X):
        distances = np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def getCost(self, X):
        labels = self.predict(X)
        distances = np.sum((X - self.centroids[labels]) ** 2, axis=1)
        wcss = np.sum(distances)
        return wcss