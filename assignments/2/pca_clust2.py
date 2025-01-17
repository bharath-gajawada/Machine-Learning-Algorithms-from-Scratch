# 6.2.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

def main():
    df = pd.read_feather('../../data/external/word-embeddings.feather')
    X = np.vstack(df['vit'].values)

    X_centered = X - np.mean(X, axis=0)

    cov_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    eigenvalues = np.sort(eigenvalues)[::-1]

    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
    plt.title('Scree Plot')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.savefig('figures/scree_plot.png')
    # plt.show()

if __name__ == "__main__":
    main()
