# 4.2.2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")

import models.gmm.gmm as gmm
import sklearn.mixture as skm

def bic_aic(X, n_components, log_likelihood):
    n_samples, n_features = X.shape
    
    n_params = n_components * (n_features * 2 + 1) - 1
    
    bic = np.log(n_samples) * n_params - 2 * log_likelihood
    aic = 2 * n_params - 2 * log_likelihood
    
    return bic, aic
    
def main():

    df = pd.read_feather('../../data/external/word-embeddings.feather')
    X = np.vstack(df['vit'].values)


    bic_values = []
    aic_values = []

    for i in range(1, 11):
        gmm_model = gmm.GMM(n_components=i)
        gmm_model.fit(X)
        log_likelihood = gmm_model.get_likelihood(X)
        bic, aic = bic_aic(X, i, log_likelihood)
        bic_values.append(bic)
        aic_values.append(aic)

    print(bic_values)
    print(aic_values)

    plt.plot(range(1, 11), bic_values, marker='o', label='BIC')
    plt.plot(range(1, 11), aic_values, marker='o', label='AIC')
    plt.title("BIC and AIC values for different K")
    plt.legend()
    plt.savefig("figures/gmm_bic_aic.png")
    # plt.show()

    best_k_bic = np.argmin(bic_values) + 1
    best_k_aic = np.argmin(aic_values) + 1

    print(f"Best K using BIC: {best_k_bic}")
    print(f"Best K using AIC: {best_k_aic}")

# [1672.9242389264155, 6012.6496663790795, 11044.14954234987, 16263.563567020006, 21574.685797187434, 26923.651918174954, 32293.325484256427, 37677.875856515435, 43068.97006289165, 48467.08316051327]
# [-1704.5527444187737, -745.6026176778478, 905.1219575812074, 2743.7606815396075, 4674.107610995296, 6642.298431271081, 8631.196696640814, 10634.971768188087, 12645.29067385256, 14662.628470762444]
# Best K using BIC: 1
# Best K using AIC: 1


    sk_bic_values = []
    sk_aic_values = []
    for i in range(1, 11):
        sk_gmm = skm.GaussianMixture(n_components=i, random_state=42)
        sk_gmm.fit(X)
        sk_bic_values.append(sk_gmm.bic(X))
        sk_aic_values.append(sk_gmm.aic(X))

    print(sk_bic_values)
    print(sk_aic_values)

    plt.plot(range(1, 11), sk_bic_values, marker='o', label='BIC')
    plt.plot(range(1, 11), sk_aic_values, marker='o', label='AIC')
    plt.title("BIC and AIC values for different K(Using sklearn)")
    plt.legend()
    plt.savefig("figures/sk_gmm_bic_aic.png")
    # plt.show()

    sk_best_k_bic = np.argmin(sk_bic_values) + 1
    sk_best_k_aic = np.argmin(sk_aic_values) + 1

    print(f"Best K using BIC: {sk_best_k_bic}")
    print(f"Best K using AIC: {sk_best_k_aic}")

# [-51980.38727810292, 428479.13643816067, 1050207.3949957252, 1724159.5125760701, 2404044.764740069, 3082983.7325744834, 3763027.4163919496, 4442360.8981017005, 5134653.681680048, 5825117.395918418]
# [-486830.548883796, -441224.4850905922, -254349.68645608716, -15251.028798801824, 229780.76344213728, 473866.27135349205, 719056.4952478982, 963536.5170345888, 1220975.8406898775, 1476586.0950051886]
# Best K using BIC: 1
# Best K using AIC: 1


if __name__ == "__main__":
    main()