import numpy as np
import matplotlib.pyplot as plt

class KDE:
    def __init__(self, kernel="gaussian", bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.data = None

    def kernel_function(self, u):
        if self.kernel == "gaussian":
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        elif self.kernel == "box":
            return np.where(np.abs(u) <= 1, 0.5, 0)
        elif self.kernel == "triangular":
            return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)

    def fit(self, data):
        self.data = np.asarray(data)

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        elif x.ndim == 0:
            x = np.array([x])[:, np.newaxis]
        n_samples, n_features = self.data.shape
        densities = np.zeros(x.shape[0])

        for i, point in enumerate(x):
            distances = (point - self.data) / self.bandwidth
            kernel_vals = self.kernel_function(distances)
            kernel_product = np.prod(kernel_vals, axis=1)
            densities[i] = np.sum(kernel_product) / (n_samples * self.bandwidth**n_features)
        
        return densities

    def visualize(self, grid_size=100, path_to_save=None):
        if self.data.shape[1] != 2:
            raise ValueError("Visualization only supported for 2D data.")

        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid = np.vstack([X.ravel(), Y.ravel()]).T

        Z = self.predict(grid).reshape(grid_size, grid_size)

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, cmap="cividis")
        plt.colorbar(label="Density")
        plt.scatter(self.data[:, 0], self.data[:, 1], color="red", s=1, label="Data points")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(f"KDE with {self.kernel} kernel")
        plt.legend()
        plt.show()
        if path_to_save:
            plt.savefig(path_to_save)

