# Laptop Price Predication Using Machine Learning

A small Python project that predicts laptop prices using machine learning. This repository contains data exploration, feature engineering, model training, and evaluation code to build a regression model that estimates laptop prices from technical specifications and features.

> Note: The repository name uses "predication" (as in the original repo name). This README assumes the project is implemented in Python and may include notebooks or scripts for training and inference.

## Table of Contents

- Overview
- Dataset
- Features
- Approach
- Results
- Requirements
- Quick Start
- Typical Commands
- Project Structure
- Notes & Tips
- Contributing
- License
- Contact

## Overview

The goal of this project is to build a model that can predict the market price of a laptop based on its specifications (brand, RAM, storage, CPU, GPU, display, weight, etc.). This is useful for price estimation in marketplaces, customer recommendation systems, or analyzing pricing trends.

## Dataset

- The dataset should include laptop specifications and price (target). If not already present, place a CSV (for example `laptops.csv`) in a `data/` directory.
- Common columns: Brand, Model, Processor, RAM, Storage, GPU, DisplaySize, Resolution, Weight, OS, Price (target).

If you used a public dataset (e.g., from Kaggle), cite it here and add the source link.

## Features

Typical features used in modeling:
- Categorical: Brand, Processor type, GPU, Operating System
- Numerical: RAM (GB), Storage (GB), DisplaySize (inches), Weight (kg)
- Engineered: Price category, CPU benchmark score, SSD vs HDD flags, pixel density (PPI)

## Approach

1. Exploratory Data Analysis (EDA)
2. Data cleaning and missing value handling
3. Feature engineering and encoding (One-Hot, Target Encoding, Ordinal where applicable)
4. Train/Test split and cross-validation
5. Train regression models (e.g., Linear Regression, RandomForestRegressor, XGBoost, LightGBM)
6. Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
7. Evaluate with metrics: RMSE, MAE, R²
8. Save best model (pickle or joblib)

## Results

Provide a short summary of the best-performing model and metrics, for example:

- Best model: XGBoost / RandomForest
- RMSE: 154.32
- MAE: 98.45
- R²: 0.87

(Replace with actual results from your runs.)

## Requirements

Create a virtual environment and install dependencies:

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn (for visualization)
- xgboost or lightgbm (optional)
- jupyterlab / notebook (optional)
- joblib

A sample requirements file (requirements.txt) can be used:
pip install -r requirements.txt

## Quick Start

1. Clone the repo:
   git clone https://github.com/pendyalagnaneswari/laptop_price_predication_using_machine_learning.git
2. Create & activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Prepare data:
   - Put your CSV dataset in the `data/` directory (e.g., `data/laptops.csv`).
5. Run EDA / Notebook:
   jupyter lab
   - Open the main notebook (if present) such as `notebooks/Laptop_Price_Prediction.ipynb`
6. Train model via script (example):
   python src/train.py --data data/laptops.csv --output models/best_model.pkl

Adjust paths and filenames to match the repository content.

## Typical Commands

- Run training (example):
  python src/train.py --data data/laptops.csv
- Evaluate:
  python src/evaluate.py --model models/best_model.pkl --test data/test.csv
- Predict single sample:
  python src/predict.py --model models/best_model.pkl --input '{"RAM":8,"Brand":"Dell",...}'

## Project Structure (recommended)

- data/                  # raw and processed datasets
- notebooks/             # EDA and experiments (Jupyter notebooks)
- src/                   # training, evaluation and inference scripts
- models/                # saved model artifacts
- requirements.txt
- README.md

Adjust to match your repository layout.

## Notes & Tips

- Standardize and scale numeric features where appropriate.
- Use cross-validation to avoid overfitting.
- Consider domain-specific feature engineering: convert display resolution to PPI, combine CPU cores & base frequency into a score, parse storage into SSD/HDD sizes.
- Log experiments (e.g., using MLflow or simple CSV logs).

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests and submit pull requests for improvements.

## License

Specify your license here (e.g., MIT). If you don't have a license, add one (LICENSE file) or state "All rights reserved".

## Contact

Author: pendyalagnaneswari  
GitHub: https://github.com/pendyalagnaneswari

```
