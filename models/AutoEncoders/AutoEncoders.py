# 4.1

import sys
sys.path.append("../..")

import models.MLP.MLPRegression as mlp

class AutoEncoder:
    
    def __init__(self, learning_rate, activation, optimizer, hidden_layers, neurons_per_layer, latent_dim, batch_size, epochs, early_stopping, patience):

        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer + [latent_dim] + neurons_per_layer[::-1]
        
        self.mlp = mlp.MLPRegressor(learning_rate, activation, optimizer, 2*hidden_layers + 1, self.neurons_per_layer, batch_size, epochs, early_stopping, patience)
        
    def fit(self, X):

        self.mlp.fit(X, X)

    def get_latent(self, X):

        _, AZ = self.mlp.forward_prop(X)
        latent = AZ[self.hidden_layers][1]
        return latent