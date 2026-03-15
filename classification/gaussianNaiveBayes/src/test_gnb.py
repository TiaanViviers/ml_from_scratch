import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import confusion_matrix, classification_report

from gaussian_nb import GaussianNB

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def main():
    df = pd.read_csv(PROJECT_ROOT / "data" / "iris.csv")
    Xt = df.drop("variety", axis=1).to_numpy().T
    y = df["variety"].to_numpy()
    print(f"Xt shape: {Xt.shape}")
    print(f"y shape: {y.shape}")
    
    gnb = GaussianNB()
    gnb.fit(Xt, y)
    log_probas = gnb.predict_log_proba(Xt)
    probas = gnb.predict_proba(Xt)
    preds = gnb.predict(Xt)
    
    cm = confusion_matrix(y, preds)
    print(cm)
    print()
    cr = classification_report(y, preds)
    print(cr)



if __name__ == "__main__":
    main()