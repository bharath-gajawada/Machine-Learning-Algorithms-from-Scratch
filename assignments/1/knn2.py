import time
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../..")

import models.knn.knn as knn


def generate_k_values(max_k):
    step_size = 0
    k_values = []
    k = 1
    while k <= max_k:
        k_values.append(k)
        if k < 20:
            step_size = 5
        elif k < 50:
            step_size = 10
        else:
            step_size = 25

        k += step_size
    return k_values

def split(X, y, test_size=0.1, val_size=0.1):
        np.random.seed(42)
        ind = np.random.permutation(len(X))
        
        test_size = int(len(X) * test_size)
        val_size = int(len(X) * val_size)
        
        test_ind = ind[:test_size]
        val_ind = ind[test_size:test_size + val_size]
        train_ind = ind[test_size + val_size:]

        X_train, X_val, X_test = X[train_ind], X[val_ind], X[test_ind]
        y_train, y_val, y_test = y[train_ind], y[val_ind], y[test_ind]

        return X_train, y_train, X_val, y_val, X_test, y_test

def main():


    df = pd.read_csv('../../data/interim/spotify_modified.csv')

    df = (df - df.mean()) / df.std()

    X = df.drop('track_genre', axis=1).to_numpy()
    y = df['track_genre'].to_numpy()


    #2.4.1.1

    max_k = (X.shape[0]) ** 0.5

    k_values = generate_k_values(max_k)

    dist_types = ['euclidean', 'manhattan', 'cosine']

    best_hyper_params = []

    X_train, y_train, X_val, y_val, X_test, y_test = split(X, y, test_size=0.1, val_size=0.1)

    for i in k_values:
        for dist_type in dist_types:

            model = knn.KNN(k=i, distance_type=dist_type)

            model.fit(X_train, y_train)

            metrics = model.validate(X_val, y_val)
            accuracy = metrics['accuracy']

            best_hyper_params.append((accuracy, i, dist_type))

    best_hyper_params.sort(reverse=True, key=lambda x: x[0])

    np.save('../../data/interim/best_hyper_params(knn).npy', best_hyper_params)

    #2.4.1.2

    for acc, k_val, dist_type in best_hyper_params[:10]:
        print(f"k: {k_val}, dist_type: {dist_type}, Accuracy: {acc}")

    #2.4.1.3

    for distance_type in dist_types:
        k_values = [k for _, k, dist_type in best_hyper_params if dist_type == distance_type]
        accuracy = [acc for acc, _, dist_type in best_hyper_params if dist_type == distance_type]

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracy, marker='o', linestyle='-', label=f'Distance: {distance_type}')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title(f'k vs Accuracy for distance type: {distance_type}')
        plt.grid(True)

        plt.savefig(f'figures/k_vs_accuracy_{distance_type}.png')
        # plt.show()

    # k: 41, dist_type: manhattan, Accuracy: 0.2349328888498991
    # k: 31, dist_type: manhattan, Accuracy: 0.23238880603561715
    # k: 51, dist_type: manhattan, Accuracy: 0.23238880603561715
    # k: 76, dist_type: manhattan, Accuracy: 0.2322133520484253
    # k: 101, dist_type: manhattan, Accuracy: 0.23080972015089044
    # k: 126, dist_type: manhattan, Accuracy: 0.2298447232213352
    # k: 151, dist_type: manhattan, Accuracy: 0.22958154224054741
    # k: 21, dist_type: manhattan, Accuracy: 0.22905518027897184
    # k: 176, dist_type: manhattan, Accuracy: 0.22844109132380033
    # k: 201, dist_type: manhattan, Accuracy: 0.22694973243266953

    #2.5

    import sklearn.neighbors as sk_knn

    X_train_sizes = [500, 1000, 5000, 10000, 50000]
    run_times_best_model = []
    run_times_sklearn_model = []

    for size in X_train_sizes:
        X_train_sample = X_train[:size]
        y_train_sample = y_train[:size]

        X_test_sample = X_test[:size//10]
        y_test_sample = y_test[:size//10]

        start_time = time.time()
        model_best = knn.KNN(k=best_hyper_params[0][1], distance_type=best_hyper_params[0][2])
        model_best.fit(X_train_sample, y_train_sample)
        metrics = model_best.predict(X_test_sample)
        end_time = time.time()
        run_times_best_model.append(end_time - start_time)

        start_time = time.time()
        model_sklearn = sk_knn.KNeighborsRegressor(n_neighbors=best_hyper_params[0][1], metric=best_hyper_params[0][2])
        model_sklearn.fit(X_train_sample, y_train_sample)
        y_pred = model_sklearn.predict(X_test_sample)
        end_time = time.time()
        run_times_sklearn_model.append(end_time - start_time)


    #2.5.1.2

    plt.figure(figsize=(10, 6))

    bar_width = 0.35
    positions = np.arange(2)

    plt.bar(positions[0], run_times_best_model[-1], width=bar_width, label='Best KNN Model')
    plt.bar(positions[1], run_times_sklearn_model[-1], width=bar_width, label='Sklearn KNN Model')

    plt.xlabel('KNN Model')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time of KNN Models')

    plt.xticks(positions + bar_width / 2, ['Best KNN Model', 'Sklearn KNN Model'])
    plt.legend()
    plt.grid(True)

    plt.savefig('figures/knn_inference_times.png')

    # plt.show()

    #2.5.1.3

    plt.figure(figsize=(10, 6))
    plt.plot(X_train_sizes, run_times_best_model, label='Best KNN Model', marker='o')
    plt.plot(X_train_sizes, run_times_sklearn_model, label='Sklearn KNN Model', marker='o')
    plt.xlabel('Training Dataset Size')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time vs Training Dataset Size')
    plt.legend()
    plt.grid(True)

    plt.savefig('figures/inference_times_vs_dataset_size(KNN).png')

    # plt.show()

if __name__ == "__main__":
    main()


