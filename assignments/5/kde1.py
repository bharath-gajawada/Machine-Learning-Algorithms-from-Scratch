import sys
sys.path.append("../..")

import models.kde.kde as kde
import models.gmm.gmm as gmm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def main():

    ## 2.2
    np.random.seed(42)

    circle_params = [
        {'num_points': 3000, 'radius': 2, 'center': (0, 0)},  # Large circle
        {'num_points': 500, 'radius': 0.5, 'center': (1, 1)}   # Small circle
    ]

    data = []

    for params in circle_params:
        num_points = params['num_points']
        radius = params['radius']
        center = params['center']
        
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        radii = radius * np.sqrt(np.random.uniform(0, 1, num_points))
        
        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)
        
        circle = np.vstack((x, y)).T
        data.append(circle)

    data = np.vstack(data)


    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], s=1, color="black", alpha=0.5)
    plt.title("Original Data")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    # plt.show()
    plt.savefig("figures/original_data(KDE).png")



    ## 2.3
    kde_model = kde.KDE(kernel="gaussian", bandwidth=0.2)

    kde_model.fit(data)
    kde_model.visualize(grid_size=100, path_to_save="figures/kde_visualization.png")



    n_components_list = [2, 5]

    for n_components in n_components_list:
        model = gmm.GMM(n_components=n_components)
        model.fit(data)

        params = model.get_params()
        centers = params['centers']
        covariances = params['covariances']
        weights = params['weights']

        # print(f"\nGMM with {n_components} Components:")
        # print("GMM Centers:\n", centers)
        # print("GMM Covariances:\n", covariances)
        # print("GMM Weights:\n", weights)

        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0], data[:, 1], s=1, color="black", alpha=0.5)

        for i in range(model.n_components):
            mean = centers[i]
            cov = covariances[i]

            x, y = np.mgrid[-4:4:.01, -4:4:.01]
            pos = np.dstack((x, y))
            rv = multivariate_normal(mean=mean, cov=cov)

            plt.contour(x, y, rv.pdf(pos), levels=5, colors=['red', 'blue', 'green', 'purple', 'orange'][i % 5], alpha=0.6)

        plt.title(f"Data and GMM Components ({n_components} Components)")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        # plt.show()
        plt.savefig(f"figures/gmm_components_{n_components}.png")


if __name__ == "__main__":
    main()
