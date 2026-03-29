import numpy as np
import matplotlib.pyplot as plt
from k_means import KMeans
from k_means_utils import (
    best_k_by_silhouette,
    cluster_sizes,
    compute_distortion,
    evaluate_k_grid,
    plot_clusters_2d,
    plot_elbow_curve,
    plot_silhouette_curve,
    silhouette_score,
)

DATA_DIR = "../../../data/"


def main():
    data = np.genfromtxt(DATA_DIR + "gaussian_spherical.csv", delimiter=",", names=True)
    X = np.column_stack((data["X1"], data["X2"]))
    Xt = X.T
    y = data["y"]

    model = KMeans(n_clusters=3, max_iter=300, tol=1e-4, random_state=42)
    labels = model.fit_predict(Xt)

    distortion = compute_distortion(Xt, labels, model.centroids)
    silhouette = silhouette_score(Xt, labels)

    print("y shape:", y.shape)
    print("labels shape:", labels.shape)
    print("model distortion:", model.distortion)
    print("utils distortion:", distortion)
    print("distortions match:", np.isclose(model.distortion, distortion))
    print("silhouette score:", silhouette)
    print("cluster sizes:", cluster_sizes(labels))

    grid_results = evaluate_k_grid(
        Xt,
        k_values=[2, 3, 4, 5],
        max_iter=300,
        tol=1e-4,
        random_state=42,
    )

    for result in grid_results:
        print(
            f"k={result['k']}, "
            f"distortion={result['distortion']:.4f}, "
            f"silhouette={result['silhouette_score']:.4f}, "
            f"n_iter={result['n_iter']}, "
            f"converged={result['converged']}"
        )

    best_result = best_k_by_silhouette(grid_results)
    print("best k by silhouette:", best_result["k"])

    plot_clusters_2d(Xt, labels, model.centroids, title="K-Means Clusters")
    plot_elbow_curve(grid_results)
    plot_silhouette_curve(grid_results)
    plt.show()
    
    
if __name__ == "__main__":
    main()
