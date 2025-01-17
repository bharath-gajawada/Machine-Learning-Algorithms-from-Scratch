# 3.2.1

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("../..")

import models.k_means.k_means as Km

def main():

    df = pd.read_feather('../../data/external/word-embeddings.feather')
    X = np.vstack(df['vit'].values)

    wcss = []
    for i in range(1,200):
        model = Km.KMeans(i+1)
        model.fit(X)
        wcss.append(model.getCost(X))

    plt.plot(range(1,200), wcss, marker='o', linestyle='-', markersize=4)
    plt.xlabel('Number of Clusters(K)')
    plt.ylabel('WCSS')
    plt.savefig('figures/wcss_vs_k.png')
    # plt.show()

if __name__ == "__main__":
    main()