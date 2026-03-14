from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pca import PCA


def main():
    df = pd.read_csv(PROJECT_ROOT / "data" / "YM_vol.csv")
    print(df.columns)
    X = df.to_numpy()
    X_t = X.T

    pca = PCA(n_components=2, whiten=True)
    pca.fit_svd(X_t)

    print(pca.principle_components)
    print(pca.explained_variance)


if __name__ == "__main__":
    main()
