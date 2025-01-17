# 3.5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

import models.MLP.binaryclassification as MLPreg

def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

def split(X, y, val_size=0.2, test_size=0.2):
    np.random.seed(42)
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_val = int(n * val_size)
    n_test = int(n * test_size)

    val_indices = indices[:n_val]
    test_indices = indices[n_val:n_val + n_test]
    train_indices = indices[n_val + n_test:]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    X_train = X[train_indices]
    y_train = y[train_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():

    data = pd.read_csv('../../data/external/diabetes.csv') 

    X = data.drop('Outcome', axis=1).values
    X = normalize(X)
    y = data['Outcome'].values
    y = y.reshape(-1, 1)

    X_train, y_train, _, _, X_test, y_test = split(X, y, val_size=0,test_size=0.2)
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T


    mse_model = MLPreg.MLPRegressor(learning_rate=0.01, activation='sigmoid', optimizer='mini_batch',hidden_layers=1, neurons_per_layer=[1], batch_size=32, epochs=1000, early_stopping=True, patience=10, loss_function="mse")

    mse_losses = mse_model.fit(X_train, y_train)

    bce_model = MLPreg.MLPRegressor(learning_rate=0.01, activation='sigmoid', optimizer='mini_batch', hidden_layers=1, neurons_per_layer=[1], batch_size=32, epochs=1000, early_stopping=True, patience=10, loss_function="bce")

    bce_losses = bce_model.fit(X_train, y_train)


    plt.figure(figsize=(10,5))
    plt.plot(mse_losses, label='MSE Loss')
    plt.title('MSE Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('figures/mse_loss.png')

    plt.figure(figsize=(10,5))
    plt.plot(bce_losses, label='BCE Loss')
    plt.title('BCE Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('figures/bce_loss.png')

if __name__ == "__main__":
    main()