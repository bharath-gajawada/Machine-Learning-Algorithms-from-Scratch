# 5.2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

import models.pca.pca as pca

def main():

    df = pd.read_feather('../../data/external/word-embeddings.feather')
    X = np.vstack(df['vit'].values)

    pca_2d = pca.PCA(n_components=2)
    pca_2d.fit(X)
    X_transformed = pca_2d.transform(X)
    print(pca_2d.checkPCA(X))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
    plt.savefig('figures/pca_2d.png')
    # plt.show()

    pca_3d = pca.PCA(n_components=3)
    pca_3d.fit(X)
    X_transformed_3 = pca_3d.transform(X)
    print(pca_3d.checkPCA(X))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_transformed_3[:, 0], X_transformed_3[:, 1], X_transformed_3[:, 2])
    plt.savefig('figures/pca_3d.png')
    # plt.show()

if __name__ == "__main__":
    main()