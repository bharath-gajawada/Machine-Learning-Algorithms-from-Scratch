#8.3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

def main():
    df = pd.read_feather('../../data/external/word-embeddings.feather')
    X = np.vstack(df['vit'].values)

    pairwise_distances = pdist(X, metric='euclidean')
    Z = linkage(pairwise_distances, method='complete')

    kbest1 = 12  # best k from K-means clustering
    kbest2 = 3   # best k from GMM clustering

    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram Kbest")
    dendrogram(Z)

    plt.axhline(y=Z[-kbest1, 2], color='r', linestyle='--', label=f'Cut for kbest1(={kbest1})')
    plt.axhline(y=Z[-kbest2, 2], color='g', linestyle='--', label=f'Cut for kbest2(={kbest2})')
    plt.legend()
    plt.savefig('figures/dendrogram_kbest.png')
    # plt.show()

    clusters_kbest1 = fcluster(Z, kbest1, criterion='maxclust')
    print(f"Labels for kbest1({kbest1}) : {clusters_kbest1}")

    # Labels for kbest1(12) : [ 2  2  9  9  9  2  4  2  2  4  1  2  9  6  7  3  9  2  2  4  2  4  4  2
    #   4  5  9  2  2  5  4  7  5  6  1  2  2  7  2  9  2  2  2  6  8  4  2  6
    #   3  2  9  2  2  2  6  6  7  2  9  2  7  6  9  2  5  2  5  1  2  5  1  1
    #   2  2 10  8 10  2  8  4  2  6  3  8  5  4  2  3  4  5  4  1  2 11  4  9
    #   2  7  5  2  7  5  5  2  2  3 12  3  4  6  2  2  2  3  8  9  1 10  9  6
    #   6  5  5  4  9  2  6  2  2  2  2  2  5  2  9  3  4  7  7  2  7  4  2  2
    #   2  7  7  2  6  2  4  4  4  2  8  3  2  2  6  8  2  8  9  6  9  8  7  2
    #   2  8  2  2  9  2  8  5  6  7  2  7  4  6  4  6  2  2  5  8  7  2  9  3
    #   4  5  7  8  8  2  5  7]

    clusters_kbest2 = fcluster(Z, kbest2, criterion='maxclust')
    print(f"Labels for kbest2({kbest2}) : {clusters_kbest2}")

    # Labels for kbest2(3) : [1 1 2 2 2 1 2 1 1 2 1 1 2 2 2 2 2 1 1 2 1 2 2 1 2 2 2 1 1 2 2 2 2 2 1 1 1
    # 2 1 2 1 1 1 2 2 2 1 2 2 1 2 1 1 1 2 2 2 1 2 1 2 2 2 1 2 1 2 1 1 2 1 1 1 1
    # 3 2 3 1 2 2 1 2 2 2 2 2 1 2 2 2 2 1 1 3 2 2 1 2 2 1 2 2 2 1 1 2 3 2 2 2 1
    # 1 1 2 2 2 1 3 2 2 2 2 2 2 2 1 2 1 1 1 1 1 2 1 2 2 2 2 2 1 2 2 1 1 1 2 2 1
    # 2 1 2 2 2 1 2 2 1 1 2 2 1 2 2 2 2 2 2 1 1 2 1 1 2 1 2 2 2 2 1 2 2 2 2 2 1
    # 1 2 2 2 1 2 2 2 2 2 2 2 1 2 2]

if __name__ == "__main__":
    main()

