import numpy as np

class KNN:
    def __init__(self, k, distance_type):
        self.k = k
        self.distance_type = distance_type
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def calculate_distance(self, X_train, x):
        if self.distance_type == 'euclidean':
            return np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        elif self.distance_type == 'manhattan':
            return np.sum(np.abs(X_train - x), axis=1)
        elif self.distance_type == 'cosine':
            x_norm = np.linalg.norm(x)
            X_train_norms = np.linalg.norm(self.X_train, axis=1)
            return 1 - np.dot(self.X_train, x) / (X_train_norms * x_norm)
        
    def predict(self, X):
        predictions = np.empty(X.shape[0], dtype=self.y_train.dtype)

        for idx, x in enumerate(X):
            distances = self.calculate_distance(self.X_train, x)
            k_nearest_ind = np.argpartition(distances, self.k)[:self.k]
            k_nearest_val = self.y_train[k_nearest_ind]
            values, counts = np.unique(k_nearest_val, return_counts=True)
            predictions[idx] = values[np.argmax(counts)]

        return predictions

    def validate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        metrics = Metrics(y_val, y_pred)
        
        return {
            'accuracy': metrics.accuracy(),
            'precision micro': metrics.precision(method='micro'),
            'precision macro': metrics.precision(method='macro'),
            'recall micro': metrics.recall(method='micro'),
            'recall macro': metrics.recall(method='macro'),
            'f1 micro': metrics.f1(method='micro'),
            'f1 macro': metrics.f1(method='macro')
        }
    
class Metrics:
    def __init__(self, y_acut, y_pred):
        self.y_acut = y_acut
        self.y_pred = y_pred
        self.confusion_matrix = self.confusion_matrix()

    def confusion_matrix(self):
        classes = np.unique(np.concatenate((self.y_acut, self.y_pred)))
        matrix = np.zeros((len(classes), len(classes)))

        for i in range(len(classes)):
            for j in range(len(classes)):
                matrix[i, j] = np.sum((self.y_acut == classes[i]) & (self.y_pred == classes[j]))

        return matrix

    def accuracy(self):
        return np.mean(self.y_acut == self.y_pred)
    
    def precision(self, method):
        matrix = self.confusion_matrix
        diagonals = np.diagonal(matrix)
        if method == 'macro':
            sum_per_class = np.sum(matrix, axis=0)
            mask = (sum_per_class != 0)
        
            precisions = np.zeros_like(sum_per_class, dtype=float)
            precisions[mask] = diagonals[mask] / sum_per_class[mask]
            return np.mean(precisions)
        elif method == 'micro':
            return np.sum(diagonals) / np.sum(matrix)
        
    def recall(self, method):
        matrix = self.confusion_matrix
        diagonals = np.diagonal(matrix)
        if method == 'macro':
            sum_per_class = np.sum(matrix, axis=1)
            mask = (sum_per_class != 0)
        
            recalls = np.zeros_like(sum_per_class, dtype=float)
            recalls[mask] = diagonals[mask] / sum_per_class[mask]
            return np.mean(recalls)
        elif method == 'micro':
            return np.sum(diagonals) / np.sum(matrix)
        
    def f1(self, method):
        precision = self.precision(method)
        recall = self.recall(method)
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)