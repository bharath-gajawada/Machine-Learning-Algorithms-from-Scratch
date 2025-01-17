# 3.2.2

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("../..")

import models.k_means.k_means as Km

def main():

    df = pd.read_feather('../../data/external/word-embeddings.feather')
    X = np.vstack(df['vit'].values)

    Kkmeans1 = 3 # Number of clusters from elbow method
    model = Km.KMeans(Kkmeans1)
    model.fit(X)

    labels = model.predict(X)
    cost = model.getCost(X)
    print(f"Labels: {labels}")
    print(f"Cost: {cost}")

# Labels: [1 1 2 2 2 1 2 1 1 2 1 1 2 2 2 2 2 1 1 2 1 2 2 1 2 2 2 1 1 2 2 2 2 2 1 1 1
#  2 1 2 1 2 1 1 2 2 1 1 1 1 2 1 1 1 2 2 2 1 2 1 2 2 2 1 2 1 2 1 1 1 1 1 1 1
#  2 2 2 2 2 1 0 2 2 2 2 1 1 2 2 2 1 1 1 2 1 0 1 2 2 1 2 2 2 2 1 2 0 2 2 2 1
#  1 2 1 2 2 1 2 2 2 2 2 2 2 2 1 2 1 1 1 1 1 2 1 2 2 1 1 2 1 2 2 1 0 1 2 2 1
#  2 1 2 2 2 1 2 2 1 1 2 2 1 2 2 2 2 2 2 2 1 0 2 1 2 1 2 2 2 1 1 2 1 2 2 2 1
#  1 2 2 2 1 2 2 1 2 2 1 1 1 2 0]
# Cost: 4227.504432396106



if __name__ == "__main__":
    main()