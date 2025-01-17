# 2.6

import numpy as np
import pandas as pd

import sys
sys.path.append("../..")

import models.MLP.MLPMultiLabelClassifier as MultiLabelClassifier

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
    df = pd.read_csv('../../data/external/advertisement.csv')


    df['gender'] = pd.factorize(df['gender'])[0]
    df['education'] = pd.factorize(df['education'])[0]
    df['city'] = pd.factorize(df['city'])[0]
    df['occupation'] = pd.factorize(df['occupation'])[0]
    df['most bought item'] = pd.factorize(df['most bought item'])[0]

    df['married'] = df['married'].astype(int)

    labels_split = df['labels'].str.get_dummies(sep=' ')

    X = df.drop(columns=['labels']).values

    X = normalize(X)
    y = labels_split.values

    X_train, y_train, _, _, X_test, y_test = split(X, y, val_size=0 ,test_size=0.2)


    mlp = MultiLabelClassifier.MultiLabelMLPClassifier(learning_rate=0.0001,activation='relu',optimizer='mini_batch',hidden_layers=2,neurons_per_layer=[64, 32],batch_size=32,epochs=100,early_stopping=True,patience=5)

    mlp.fit(X_train.T, y_train.T)
    y_pred = mlp.predict(X_test.T)
    metrics = mlp.evaluate(X_test.T, y_test.T)
    print(metrics)

    # {'Accuracy': 0.01, 'Precision': 0.12573195187165775, 'Recall': 0.36363636363636365, 'F1-Score': 0.18685602775099358, 'Hamming Loss': 0.45}

if __name__ == "__main__":
    main()

