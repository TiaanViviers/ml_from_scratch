# LDA From Scratch

Linear Discriminant Analysis implementation for dimensionality reduction (CS315 practice).

## Files

- `src/lda.py`: LDA implementation
- `src/lda_utils.py`: helper utilities and plotting
- `src/test_lda.py`: quick script using `data/iris.csv`
- `comparison/LDA_comparison.ipynb`: scratch vs sklearn notebook

## Data convention

This project uses:
- rows = features
- columns = observations

So `X.shape == (d, N)`.

## Quick run

- From repo root:
  - `python preprocessing/lda/src/test_lda.py`
