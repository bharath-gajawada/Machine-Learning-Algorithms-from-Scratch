# 2.2

import numpy as np


class MLPClassifier:
    def __init__(self, learning_rate, activation, optimizer, hidden_layers, neurons_per_layer, batch_size, epochs, early_stopping, patience):
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience

        self.weights = []
        self.biases = []

        self.best_loss = float('inf')
        self.patience_counter = 0

    def init_params(self, input_neurons, output_neurons):
        np.random.seed(42)
        self.weights.append(np.random.randn(self.neurons_per_layer[0], input_neurons) * np.sqrt(2 / input_neurons))
        self.biases.append(np.zeros((self.neurons_per_layer[0], 1)))

        for i in range(1, self.hidden_layers):
            self.weights.append(np.random.randn(self.neurons_per_layer[i], self.neurons_per_layer[i - 1]) * np.sqrt(2 / self.neurons_per_layer[i - 1]))
            self.biases.append(np.zeros((self.neurons_per_layer[i], 1)))

        self.weights.append(np.random.randn(output_neurons, self.neurons_per_layer[-1]) * np.sqrt(2 / self.neurons_per_layer[-1]))
        self.biases.append(np.zeros((output_neurons, 1)))
                            

    def activation_function(self, activation, Z, derivative=False):
        if activation == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
            if derivative:
                return A * (1 - A)
            return A
        elif activation == 'tanh':
            if derivative:
                return 1 - np.tanh(Z) ** 2
            return np.tanh(Z)
        elif activation == 'relu':
            if derivative:
                return np.where(Z > 0, 1, 0)
            return np.maximum(0, Z)
        elif activation == 'linear':
            if derivative:
                return 1
            return Z
        elif activation == 'softmax':
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return expZ / expZ.sum(axis=0, keepdims=True)


    def forward_prop(self, X):
        A = X
        AZ = []
        for i in range(self.hidden_layers):
            Z = np.dot(self.weights[i], A) + self.biases[i]
            AZ.append((A, Z))
            A = self.activation_function(self.activation, Z)
        Z = np.dot(self.weights[-1], A) + self.biases[-1]
        AZ.append((A, Z))
        A = self.activation_function('softmax', Z)
        return A, AZ
    
    def back_prop(self, y, A_final, AZ):

        m = y.shape[1]

        dA = A_final - y

        dW = []
        db = []

        for i in reversed(range(self.hidden_layers + 1)):

            A_prev, Z = AZ[i]

            if i == self.hidden_layers:
                dZ = dA
            else:
                dZ = dA * self.activation_function(self.activation, Z, derivative=True)

            dW.append((1 / m) * np.dot(dZ, A_prev.T))
            db.append((1 / m) * np.sum(dZ, axis=1, keepdims=True))

            dA = np.dot(self.weights[i].T, dZ)

        dW.reverse()
        db.reverse()

        return dW, db
    
    def update_weights(self, dW, db):
        for i in range(self.hidden_layers + 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def cross_entropy(self, y, A_final):
        return -np.sum(y * np.log(A_final + 1e-8)) / y.shape[1]

    def fit(self, X, y):
        input_neurons = X.shape[0]
        output_neurons = y.shape[0]

        self.init_params(input_neurons, output_neurons)

        batch_size = 0
        if self.optimizer == "sgd":
            batch_size = 1
        elif self.optimizer == "batch":
            batch_size = X.shape[1]
        elif self.optimizer == "mini_batch":
            batch_size = self.batch_size
        
        for epoch in range(self.epochs):
            for i in range(0, X.shape[1], batch_size):
                X_batch = X[:, i:i + batch_size]
                y_batch = y[:, i:i + batch_size]

                A_final, AZ = self.forward_prop(X_batch)
                dW, db = self.back_prop(y_batch, A_final, AZ)
                self.update_weights(dW, db)

            if self.early_stopping:
                A_final, _ = self.forward_prop(X)
                loss = self.cross_entropy(y, A_final)

                if loss < self.best_loss:
                    self.best_loss = loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter == self.patience:
                    break


    def predict(self, X):
        A_final, _ = self.forward_prop(X)
        pred = np.argmax(A_final, axis=0)
        return pred
    
    
    def numerical_gradient(self, X, y, epsilon=1e-7):
        dW = []
        dB = []

        for i in range(self.hidden_layers + 1):
            dW.append(np.zeros(self.weights[i].shape))
            dB.append(np.zeros(self.biases[i].shape))

        for i in range(self.hidden_layers + 1):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    self.weights[i][j, k] += epsilon
                    A_final, _ = self.forward_prop(X)
                    loss1 = self.cross_entropy(y, A_final)

                    self.weights[i][j, k] -= 2 * epsilon
                    A_final, _ = self.forward_prop(X)
                    loss2 = self.cross_entropy(y, A_final)

                    dW[i][j, k] = (loss1 - loss2) / (2 * epsilon)
                    self.weights[i][j, k] += epsilon

            for j in range(self.biases[i].shape[0]):
                self.biases[i][j, 0] += epsilon
                A_final, _ = self.forward_prop(X)
                loss1 = self.cross_entropy(y, A_final)

                self.biases[i][j, 0] -= 2 * epsilon
                A_final, _ = self.forward_prop(X)
                loss2 = self.cross_entropy(y, A_final)

                dB[i][j, 0] = (loss1 - loss2) / (2 * epsilon)
                self.biases[i][j, 0] += epsilon

        return dW, dB

    def gradient_check(self, X, y, epsilon=1e-7, threshold=1e-5):
        A_final, AZ = self.forward_prop(X)
        comp_dW, comp_dB = self.back_prop(y, A_final, AZ)
        num_dW, num_dB = self.numerical_gradient(X, y, epsilon)

        passed = 0

        for i in range(self.hidden_layers + 1):
            print(f"Layer {i + 1}")
            difference = np.linalg.norm(comp_dW[i] - num_dW[i])/np.linalg.norm(comp_dW[i] + num_dW[i] + epsilon)
            print(f"Weight difference: {difference}")
            if difference < threshold:
                passed += 1


            difference = np.linalg.norm(comp_dB[i] - num_dB[i])/np.linalg.norm(comp_dB[i] + num_dB[i] + epsilon)
            print(f"Bias difference: {difference}")
            if difference < threshold:
                passed += 1

        return passed/(2 * (self.hidden_layers + 1))