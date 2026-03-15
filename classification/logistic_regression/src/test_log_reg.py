import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, log_loss

from logistic_regression import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
# Resolve repo root robustly.
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def main():
    df = pd.read_csv(PROJECT_ROOT / "data" / "adult.csv")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    # Standardize using train statistics only (no leakage).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_t = X_train.T
    X_val_t = X_val.T
    X_test_t = X_test.T

    print("DATA SHAPES:")
    print(f"    X_train_t shape: {X_train_t.shape}")
    print(f"    y_train shape: {y_train.shape}\n")
    print(f"    X_val_t shape: {X_val_t.shape}")
    print(f"    y_val shape: {y_val.shape}\n")
    print(f"    X_test_t shape: {X_test_t.shape}")
    print(f"    y_test shape: {y_test.shape}\n")
    
    print("CLASS BALANCE:")
    print_class_balance("train", y_train)
    print_class_balance("val", y_val)
    print_class_balance("test", y_test)
    print("")

    lambda_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = None
    best_val_logloss = np.inf
    best_model = None

    print("VALIDATION SEARCH:")
    for lam in lambda_grid:
        model = LogisticRegression(lambda_reg=lam, max_iter=100, tol=1e-6, reg_bias=True)
        model.fit(X_train_t, y_train)

        val_preds = model.predict(X_val_t)
        val_proba = model.predict_proba(X_val_t).T  # (n_samples, n_classes)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, pos_label=model.class_labels[1])
        val_logloss = log_loss(y_val, val_proba, labels=list(model.class_labels))

        print(
            f"  lambda={lam:>7}: "
            f"val_logloss={val_logloss:.5f}, "
            f"val_acc={val_acc:.4f}, "
            f"val_f1={val_f1:.4f}, "
            f"iters={model.n_iter}"
        )

        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            best_lambda = lam
            best_model = model

    print(f"\nBest lambda from validation: {best_lambda} (logloss={best_val_logloss:.5f})")

    # Refit on train+val with best lambda, then evaluate once on test.
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    X_trainval_t = X_trainval.T

    final_model = LogisticRegression(
        lambda_reg=best_lambda,
        max_iter=100,
        tol=1e-6,
        reg_bias=True
    )
    final_model.fit(X_trainval_t, y_trainval)

    test_preds = final_model.predict(X_test_t)
    test_proba = final_model.predict_proba(X_test_t).T
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, pos_label=final_model.class_labels[1])
    test_logloss = log_loss(y_test, test_proba, labels=list(final_model.class_labels))

    print("\nTEST PERFORMANCE (after lambda tuning):")
    print(f"  accuracy : {test_acc:.4f}")
    print(f"  f1-score : {test_f1:.4f}")
    print(f"  log-loss : {test_logloss:.5f}")
    print(f"  iterations: {final_model.n_iter}")

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, test_preds, labels=final_model.class_labels))

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, test_preds))
    
    

def split_data(df):
    X = df.drop("incomes", axis=1)
    y = df["incomes"].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train
    )
    
    return X_train.to_numpy(dtype=float), y_train, X_val.to_numpy(dtype=float), y_val, X_test.to_numpy(dtype=float), y_test


def print_class_balance(name, y):
    vals, counts = np.unique(y, return_counts=True)
    total = len(y)
    desc = ", ".join([f"class {v}: {c} ({c/total:.2%})" for v, c in zip(vals, counts)])
    print(f"  {name:>5} -> {desc}")



if __name__ == "__main__":
    main()
