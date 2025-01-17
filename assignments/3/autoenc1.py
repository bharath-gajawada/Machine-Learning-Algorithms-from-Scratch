import numpy as np
import pandas as pd

import sys
sys.path.append("../..")

import models.AutoEncoders.AutoEncoders as AE
from  models.knn.knn import KNN, Metrics
import models.pca.pca as PCA
import models.MLP.MLPClassifier as MLPclas

def split(X, y, val_size=0.2):
        np.random.seed(42)
        ind = np.random.permutation(len(X))
        
        val_size = int(len(X) * val_size)
        
        val_ind = ind[:val_size]
        train_ind = ind[val_size:]

        X_train, X_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]

        return X_train, y_train, X_val, y_val

def onehot_encode(y):
    unique_classes = np.unique(y)
    one_hot_encoded = np.zeros((y.size, unique_classes.size))
    for i, value in enumerate(y):
        one_hot_encoded[i, np.where(unique_classes == value)] = 1
    
    return one_hot_encoded.T

def get_metrics(X, y):
    X_train, y_train, X_val, y_val = split(X, y, val_size=0.2)
    
    # best metric k: 41, dist_type: manhattan
    model = KNN(k=41, distance_type='manhattan')
    model.fit(X_train, y_train)

    metrics = model.validate(X_val, y_val)
    return metrics
     

def main():

    #4.1 4.2
    df = pd.read_csv('../../data/interim/spotify_modified.csv')

    df = (df - df.mean()) / df.std()

    X = df.drop('track_genre', axis=1).values
    y = df['track_genre'].values
    
    optimal_dim = 2 # from assignment 2


    autoencoder = AE.AutoEncoder(learning_rate=0.01, activation='sigmoid', optimizer='sgd', hidden_layers=1, neurons_per_layer=[10], latent_dim=optimal_dim, batch_size=32, epochs=100, early_stopping=True, patience=5)
    autoencoder.fit(X.T)

    latent_X = autoencoder.get_latent(X.T).T

    # print(latent_X)
    print(latent_X.shape)


    # 4.3
    latent_metrics = get_metrics(latent_X, y)
    print("Metrics of Reduced data using AutoEncoder:\n",latent_metrics)

    pca = PCA.PCA(n_components=2)
    pca.fit(X)

    reduced_X = pca.transform(X)

    reduced_metrics = get_metrics(reduced_X, y)
    print("Metrics of Reduced data using PCA:\n",reduced_metrics)


    original_metric = get_metrics(X, y)
    print("Metrics of original data:\n",original_metric)

# Metrics of Reduced data using AutoEncoder:
#  {'accuracy': 0.06434492740909689, 'precision micro': 0.06434492740909689, 'precision macro': 0.055954599023951855, 'recall micro': 0.06434492740909689, 'recall macro': 0.0644567088929016, 'f1 micro': 0.06434492740909689, 'f1 macro': 0.05990549165027538}
# Metrics of Reduced data using PCA:
#  {'accuracy': 0.06789771481205316, 'precision micro': 0.06789771481205316, 'precision macro': 0.05876470907773867, 'recall micro': 0.06789771481205316, 'recall macro': 0.06761383466489909, 'f1 micro': 0.06789771481205316, 'f1 macro': 0.06287946048507279}
# Metrics of original data:
#  {'accuracy': 0.23689635510329402, 'precision micro': 0.23689635510329402, 'precision macro': 0.22335991249114076, 'recall micro': 0.23689635510329402, 'recall macro': 0.2377190896045506, 'f1 micro': 0.23689635510329402, 'f1 macro': 0.23031591033298218}

    # 4.4

    mlp = MLPclas.MLPClassifier(learning_rate=0.01, activation='sigmoid', optimizer='sgd', hidden_layers=1, neurons_per_layer=[10], batch_size=32, epochs=100, early_stopping=True, patience=5)

    onehot_y = onehot_encode(y)

    mlp.fit(X.T, onehot_y)

    unique_values = np.unique(y)
    y_pred = unique_values[mlp.predict(X.T)]

    MLP_Metrics = Metrics(y, y_pred)
    mlp_metrics = {
            'accuracy': MLP_Metrics.accuracy(),
            'precision micro': MLP_Metrics.precision(method='micro'),
            'precision macro': MLP_Metrics.precision(method='macro'),
            'recall micro': MLP_Metrics.recall(method='micro'),
            'recall macro': MLP_Metrics.recall(method='macro'),
            'f1 micro': MLP_Metrics.f1(method='micro'),
            'f1 macro': MLP_Metrics.f1(method='macro')
        }
    
    print(mlp_metrics)

    # {'accuracy': 0.014640479302450021, 'precision micro': 0.014640479302450021, 'precision macro': 0.009085131573141681, 'recall micro': 0.014640479302450021, 'recall macro': 0.014640350877192983, 'f1 micro': 0.014640479302450021, 'f1 macro': 0.011212375914773635}






if __name__ == "__main__":
    main()


