import numpy as np
import matplotlib.pyplot as plt
from gmm import GMM
from gmm_utils import (
    best_n_components_by_bic,
    component_proportions,
    component_sizes,
    compute_aic,
    compute_bic,
    evaluate_component_grid,
    hard_assignments,
    plot_gmm_clusters_2d,
    plot_gmm_density_2d,
    plot_gmm_feature_density,
    plot_gmm_responsibilities_2d,
)

DATA_DIR = "../../../data/"


def main():
    data = np.genfromtxt(DATA_DIR + "gaussian_elliptical.csv", delimiter=",", names=True)
    X = np.column_stack((data["X1"], data["X2"]))
    Xt = X.T
    y = data["y"]

    model = GMM(
        n_components=3,
        max_iter=200,
        tol=1e-4,
        reg_covar=1e-6,
        random_state=42
    )

    model.fit(Xt)
    predicted_labels = model.predict(Xt)
    predicted_proba = model.predict_proba(Xt)

    print("y shape:", y.shape)
    print("predicted label shape:", predicted_labels.shape)
    print("predicted proba shape:", predicted_proba.shape)
    print("log_likelihood:", model.log_likelihood_)
    print("aic:", compute_aic(model, Xt))
    print("bic:", compute_bic(model, Xt))
    print("component sizes:", component_sizes(model.responsibilities_))
    print("component proportions:", component_proportions(model.responsibilities_))
    print(
        "hard assignments match model labels:",
        np.array_equal(hard_assignments(model.responsibilities_), model.labels_)
    )

    grid_results = evaluate_component_grid(
        Xt,
        component_values=[2, 3, 4, 5],
        max_iter=200,
        tol=1e-4,
        reg_covar=1e-6,
        random_state=42,
    )

    for result in grid_results:
        print(
            f"n_components={result['n_components']}, "
            f"log_likelihood={result['log_likelihood']:.4f}, "
            f"aic={result['aic']:.4f}, "
            f"bic={result['bic']:.4f}, "
            f"n_iter={result['n_iter']}, "
            f"converged={result['converged']}"
        )

    best_result = best_n_components_by_bic(grid_results)
    print("best n_components by BIC:", best_result["n_components"])

    plot_gmm_clusters_2d(
        Xt,
        responsibilities=model.responsibilities_,
        means=model.means_,
        covariances=model.covariances_,
        title="GMM Hard Assignments",
    )
    plot_gmm_responsibilities_2d(
        Xt,
        model.responsibilities_,
        title_prefix="GMM soft assignments",
    )
    plot_gmm_density_2d(
        Xt,
        model.means_,
        model.covariances_,
        weights=model.weights_,
        title="GMM Density Contours",
    )
    plot_gmm_feature_density(
        Xt,
        model.means_,
        model.covariances_,
        model.weights_,
        feature_index=0,
        title="GMM Feature 0 Density",
    )
    plt.show()


if __name__ == "__main__":
    main()
