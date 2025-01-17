# 9.1.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

def main():
    df = pd.read_csv('../../data/interim/spotify_modified.csv')

    df = (df - df.mean()) / df.std()

    X = df.drop('track_genre', axis=1).to_numpy()
    y = df['track_genre'].to_numpy()

    X_centered = X - np.mean(X, axis=0)

    cov_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    eigenvalues = np.sort(eigenvalues)[::-1]

    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.savefig('figures/scree_plot(spotify).png')
    # plt.show()

if __name__ == "__main__":
    main()
