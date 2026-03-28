import numpy as np
from k_means import KMeans

DATA_DIR = "../../../data/"


def main():
    data = np.genfromtxt(DATA_DIR + "gaussian_spherical.csv", delimiter=",", names=True)
    X = np.column_stack((data["X1"], data["X2"]))
    Xt = X.T
    y = data["y"]

    model = KMeans(n_clusters=3, max_iter=300, tol=1e-4, random_state=42)
    labels = model.fit_predict(Xt)
    y_new = y+1
    
    acc = float(np.mean(y_new == labels))
    print(acc)
    
    
if __name__ == "__main__":
    main()
