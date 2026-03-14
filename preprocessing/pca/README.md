# PCA From Scratch

Simple PCA implementation used for CS315 practice.

## Files

- `src/pca.py`: PCA class (`fit_svd`, `fit_eigh`, `transform`, `inverse_transform`)
- `src/pca_utils.py`: helper metrics and plotting
- `src/test_pca.py`: quick script using `data/YM_vol.csv`
- `comparison/sklearn_comparison.ipynb`: comparison notebook

## Data convention

This project uses:
- rows = features
- columns = observations

So `X.shape == (d, N)`.

## Quick run

- From repo root:
  - `python preprocessing/pca/src/test_pca.py`
