# Copilot Instructions for From-Scratch-Machine-Learning-Libraries

## Project Overview
This repository contains machine learning algorithms implemented from scratch, focusing on educational clarity and transparency. The two main components are:

- **KNearestClassifier/**: Implements K-Nearest Neighbors (KNN) for both regression and classification. Main logic is in `KNearestNeighbors.py`.
- **LinearRegression/**: Implements gradient descent-based linear regression. Main logic is in `MyLinearModel.py`.

## Directory Structure
- `KNearestClassifier/` and `LinearRegression/` each have their own `README.md` and core implementation files.
- `Test KNearestClassifier/` and `Test KNearestRegressor/` contain test scripts and sample data for KNN models.
- `Testing/` under `LinearRegression/` contains test scripts and sample data for linear regression.

## Key Patterns & Conventions
- **No external ML libraries**: All algorithms are implemented from scratch, without using scikit-learn, TensorFlow, etc.
- **Testing**: Test scripts are plain Python files (not using pytest/unittest). They are located in subfolders named `Test*` or `Testing/` and use sample CSVs for input data.
- **Data**: Test data is stored as CSV files alongside test scripts.
- **Model API**: Models expose fit/predict methods similar to scikit-learn, but with custom logic and attributes (e.g., `coef_`, `intercept_`).
- **Stopping Criteria**: Linear regression uses a precision-based stopping rule for gradient descent, as described in its README.

## Developer Workflows
- **Run tests**: Execute test scripts directly, e.g.:
  ```bash
  python KNearestClassifier/Test\ KNearestClassifier/TestPredict.py
  python LinearRegression/Testing/TestModule.py
  ```
- **Add new models**: Place new algorithm implementations in their own subfolders, following the pattern of existing modules.
- **Add tests**: Place test scripts and data in a `Test*` or `Testing/` subfolder within the relevant module.

## Examples
- KNN usage: See `KNearestClassifier/KNearestNeighbors.py` and `Test KNearestClassifier/TestPredict.py`.
- Linear regression usage: See `LinearRegression/MyLinearModel.py` and `Testing/TestModule.py`.

## Special Notes
- No package manager or requirements file; all code is pure Python.
- No formal test runner; tests are run as scripts.
- Keep code readable and educational—prioritize clarity over performance or brevity.
