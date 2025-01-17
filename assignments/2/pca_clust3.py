# 6.2.2 & 6.4.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

import models.k_means.k_means as Km
import models.pca.pca as PCA
import models.gmm.gmm as GMM

def bic_aic(X, n_components, log_likelihood):
    n_samples, n_features = X.shape
    
    n_params = n_components * (n_features * 2 + 1) - 1
    
    bic = np.log(n_samples) * n_params - 2 * log_likelihood
    aic = 2 * n_params - 2 * log_likelihood
    
    return bic, aic

def main():
    df = pd.read_feather('../../data/external/word-embeddings.feather')
    X = np.vstack(df['vit'].values)

    pca = PCA.PCA(n_components=6)
    pca.fit(X)
    reduced_dataset = pca.transform(X)

    wcss = []
    for i in range(1,200):
        model = Km.KMeans(i+1)
        model.fit(reduced_dataset)
        wcss.append(model.getCost(reduced_dataset))

    plt.figure(figsize=(10, 7))
    plt.plot(range(1,200), wcss, marker='o', linestyle='-', markersize=4)
    plt.xlabel('Number of Clusters(K)')
    plt.ylabel('WCSS')
    plt.savefig('figures/wcss_vs_k(reduced).png')
    # plt.show()

    bic_values = []
    aic_values = []

    for i in range(1, 11):
        gmm = GMM.GMM(n_components=i)
        gmm.fit(X)
        log_likelihood = gmm.get_likelihood(X)
        bic, aic = bic_aic(X, i, log_likelihood)
        bic_values.append(bic)
        aic_values.append(aic)

    print(bic_values)
    print(aic_values)

    plt.figure(figsize=(10, 7))
    plt.plot(range(1, 11), bic_values, marker='o', label='BIC')
    plt.plot(range(1, 11), aic_values, marker='o', label='AIC')
    plt.title("BIC and AIC values for different K(reduced)")
    plt.legend()
    plt.savefig("figures/gmm_bic_aic(reduced).png")
    # plt.show()

    best_k_bic = np.argmin(bic_values) + 1
    best_k_aic = np.argmin(aic_values) + 1

    print(f"Best K using BIC: {best_k_bic}")
    print(f"Best K using AIC: {best_k_aic}")

    # [1672.9242389264155, 6012.6496663790795, 11044.14954234987, 16263.563567020006, 21574.685797187434, 26923.651918174954, 32293.325484256427, 37677.875856515435, 43068.97006289165, 48467.08316051327]
    # [-1704.5527444187737, -745.6026176778478, 905.1219575812074, 2743.7606815396075, 4674.107610995296, 6642.298431271081, 8631.196696640814, 10634.971768188087, 12645.29067385256, 14662.628470762444]
    # Best K using BIC: 1
    # Best K using AIC: 1

if __name__ == "__main__":
    main()
