import numpy as np
from gmm import GMM

DATA_DIR = "../../../data/"


def main():
    data = np.genfromtxt(DATA_DIR + "gaussian_spherical.csv", delimiter=",", names=True)
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
    
    acc = float(np.mean(y == predicted_labels))
    print(acc)
    


if __name__ == "__main__":
    main()
