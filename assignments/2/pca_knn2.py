# 9.2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("../..")

import models.knn.knn as KNN
import models.pca.pca as PCA

def split(X, y, val_size=0.2):
        np.random.seed(42)
        ind = np.random.permutation(len(X))
        
        val_size = int(len(X) * val_size)
        
        val_ind = ind[:val_size]
        train_ind = ind[val_size:]

        X_train, X_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]

        return X_train, y_train, X_val, y_val

def main():
    df = pd.read_csv('../../data/interim/spotify_modified.csv')

    df = (df - df.mean()) / df.std()

    X = df.drop('track_genre', axis=1).to_numpy()
    y = df['track_genre'].to_numpy()

    pca = PCA.PCA(n_components=2)
    pca.fit(X)

    reduced_X = pca.transform(X)

    inference_time = []


    start_time = time.time()
    rX_train, ry_train, rX_val, ry_val = split(reduced_X, y, val_size=0.2)
    
    # best metric k: 41, dist_type: manhattan
    model = KNN.KNN(k=41, distance_type='manhattan')
    model.fit(rX_train, ry_train)

    metrics = model.validate(rX_val, ry_val)
    print("Metrics of Reduced data :\n",metrics)
    end_time = time.time()

    inference_time.append(end_time - start_time)

    # Metrics of Reduced data :
    # {'accuracy': 0.06789771481205316, 'precision micro': 0.06789771481205316, 'precision macro': 0.05876470907773867,
    #  'recall micro': 0.06789771481205316, 'recall macro': 0.06761383466489909, 'f1 micro': 0.06789771481205316, 'f1 macro': 0.06287946048507279}


    start_time = time.time()
    X_train, y_train, X_val, y_val = split(X, y, val_size=0.2)
    
    # best metric k: 41, dist_type: manhattan
    model = KNN.KNN(k=41, distance_type='manhattan')
    model.fit(X_train, y_train)

    metrics = model.validate(X_val, y_val)
    print("Metrics of Original data :\n",metrics)
    end_time = time.time()

    inference_time.append(end_time - start_time)

    # Metrics of Original data :
    #  {'accuracy': 0.23689635510329402, 'precision micro': 0.23689635510329402, 'precision macro': 0.22335991249114076,
    #  'recall micro': 0.23689635510329402, 'recall macro': 0.2377190896045506, 'f1 micro': 0.23689635510329402, 'f1 macro': 0.23031591033298218}    

    bar_width = 0.35
    positions = np.arange(2)

    plt.bar(positions[0], inference_time[0], width=bar_width, label='Reduced Data')
    plt.bar(positions[1], inference_time[1], width=bar_width, label='Original Data')

    plt.xlabel('Data')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time of KNN Models on Reduced and Original Data')

    plt.xticks(positions, ['Reduced Data', 'Original Data'])
    plt.legend()
    plt.grid(True)

    plt.savefig('figures/knn_inference_times.png') 
    # plt.show()   

if __name__ == "__main__":
    main()