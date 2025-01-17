import numpy as np

class LinearRegression:
    def __init__(self, k=1, reg_type=None, reg_lambda=0, learning_rate=0.001, iterations=1000):
        self.k = k
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None

    def polynomials(self, X):
        X_poly = np.vstack([X**i for i in range(self.k + 1)]).T
        return X_poly
    
    def fit(self, X, y):
        X_poly = self.polynomials(X)
        np.random.seed(42)
        self.coefficients = np.random.randn(self.k + 1)

        for _ in range(self.iterations):
            y_pred = np.dot(X_poly, self.coefficients)
            gradient = X_poly.T.dot(y_pred - y) / len(y)
            
            if self.reg_type == 'L1':
                gradient += self.reg_lambda * np.sign(self.coefficients)
            elif self.reg_type == 'L2':
                gradient += self.reg_lambda * self.coefficients # * 2 neglected

            self.coefficients -= self.learning_rate * gradient

    def predict(self, X):
        X_poly = self.polynomials(X)
        return np.dot(X_poly, self.coefficients)
    
    def validate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        metrics = Metrics(y_val, y_pred)
        
        return {
            'mse': metrics.mse(y_val, y_pred),
            'variance': metrics.variance(y_pred),
            'std_dev': metrics.std_dev(y_pred)
        }

class Metrics:
    def __init__(self, y_acut, y_pred):
        self.y_acut = y_acut
        self.y_pred = y_pred

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def variance(self, y_pred):
        return np.var(y_pred)
    
    def std_dev(self, y_pred):
        return np.std(y_pred)