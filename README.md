# ML From Scratch

Personal learning repo for building ML tools from scratch in Python.
This is intentionally simple and focused on understanding fundamental machine learning concepts, mostly dealt with in my Machine Learning
and Data Science modules at Stellenbosch university.

## What this repo is

- A notebook/code playground for learning.
- Not production code.
- No optimisation-heavy or deployment-oriented setup.

## Project layout

- `classification/`
  - `gaussianNaiveBayes/` (WIP)
  - `logistic_regression/` (WIP)
- `preprocessing/`
  - `pca/` (implemented)
  - `lda/` (implemented)
- `feature_selection/`
  - `stepwise_regression_selection/` (implemented)
- `data/`
  - shared datasets used by scripts/notebooks

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run module scripts directly, for example:
   - `python preprocessing/pca/src/test_pca.py`
   - `python preprocessing/lda/src/test_lda.py`
   - `python feature_selection/stepwise_regression_selection/test_stepper.py`

