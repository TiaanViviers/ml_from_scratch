"""Quick manual test for the LDA dimensionality-reduction implementation.

This script follows the CS315 data convention:
- rows represent features
- columns represent observations
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lda import LDA
from lda_utils import explained_variance_summary, plot_before_after_lda, plot_class_scatter_before_lda, plot_lda_projection


def main():
    """Load Iris data, reshape to module convention, and run LDA.

    Parameters:
    -------------------
    None

    Returns:
    --------------
    None
    """
    df = pd.read_csv(PROJECT_ROOT / "data" / "iris.csv")
    X = df.drop("variety", axis=1).to_numpy()
    X_t = X.T
    y = df["variety"].to_numpy()

    print(f"X_t shape: {X_t.shape}")
    print(f"y shape: {y.shape}")

    test_lda = LDA()
    Z = test_lda.fit_transform(X_t, y)
    summary = explained_variance_summary(test_lda.explained_variance_ratio)

    print("Explained variance ratio:", summary["explained_variance_ratio"])
    print("Cumulative variance:", summary["cumulative_explained_variance"])
    print(f"Projected shape: {Z.shape}")

    feature_names = [col for col in df.columns if col != "variety"]
    plot_class_scatter_before_lda(X_t, y, feature_indices=(0,1), feature_names=feature_names)
    plot_class_scatter_before_lda(X_t, y, feature_indices=(2,3), feature_names=feature_names)
    plot_lda_projection(Z, y)
    plt.show()



if __name__ == "__main__":
    main()
