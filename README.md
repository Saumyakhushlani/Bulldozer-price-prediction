# 🚜 Bulldozer Sale Price Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

A complete machine learning project to predict the sale price of bulldozers using historical auction data. This repository contains the data exploration, feature engineering, model training, and evaluation pipeline.

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Problem Statement](#-problem-statement)
3. [Methodology](#-methodology)
4. [Results](#-results)
5. [Feature Importance](#-feature-importance)
6. [Technologies Used](#-technologies-used)
7. [How to Run](#-how-to-run)
8. [License](#-license)

## 📖 Overview

This project uses machine learning to predict the sale price of heavy equipment at auction. Based on the dataset from the [Kaggle Bluebook for Bulldozers competition](https://www.kaggle.com/c/bluebook-for-bulldozers), the goal is to build a regression model that can accurately estimate the final price of a piece of equipment based on its usage, configuration, and historical sale data.

The project demonstrates a full data science workflow, from initial data cleaning and feature engineering to hyperparameter tuning and final model deployment.

## 🎯 Problem Statement

The primary objective is to predict the `SalePrice` of bulldozers. The model's performance is evaluated using the **Root Mean Squared Log Error (RMSLE)**, which is sensitive to underprediction and is a standard metric for price prediction tasks. This is a regression task on time series data, where the date of sale is a critical predictive component.

## 🛠️ Methodology

The project followed a structured machine learning pipeline:

1.  **Data Exploration (EDA):** The initial dataset was loaded and analyzed to understand its structure, identify missing values, and examine the data types of various features.

2.  **Feature Engineering:**
    * The `saledate` column was parsed from a string into a `datetime` object.
    * Time-based features were extracted from the sale date, including `saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek`, and `saleDayOfYear`, to capture temporal patterns.

3.  **Data Preprocessing:**
    * **Categorical Features:** All non-numeric columns were converted into numerical representations using pandas' `category` data type.
    * **Missing Values:** A robust imputation strategy was implemented:
        * For **numerical columns**, missing values were filled with the median.
        * For all columns with missing data, a binary `_is_missing` column was created to retain the information that a value was originally absent.

4.  **Modeling & Validation:**
    * **Model:** A `RandomForestRegressor` was chosen for its ability to handle complex non-linear relationships and its robustness.
    * **Validation:** A **time-based validation set** was created. The training data included sales up to the end of 2011, while the validation data consisted of all sales from 2012. This simulates a real-world scenario of predicting future prices.

5.  **Hyperparameter Tuning:**
    * `RandomizedSearchCV` was used to efficiently search for the optimal hyperparameters for the Random Forest model, focusing on parameters like `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`.

## 📈 Results

The final tuned model, trained on the full training dataset, demonstrated strong predictive performance on the unseen validation set.

| Metric         | Training Set | **Validation Set** |
| -------------- | ------------ | ------------------ |
| **RMSLE** | 0.171        | **0.245** |
| **MAE** | 3,518.99     | **5,961.26** |
| **R² Score** | 0.940        | **0.879** |


## ✨ Feature Importance

The feature importance plot reveals which factors are most influential in determining a bulldozer's sale price.

[![feature-importance.png](https://i.postimg.cc/Wz3wdjnS/feature-importance.png)](https://postimg.cc/N93Hn3Zr)

The top 5 most important features were:
1.  **YearMade:** The age of the equipment is the single most critical factor.
2.  **ProductSize:** The size category of the bulldozer.
3.  **fiProductClassDesc:** The detailed description of the machine's class.
4.  **saleYear:** The year of the auction, capturing market trends.
5.  **fiSecondaryDesc:** Additional descriptive information about the model.

## 🚀 Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-313131?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3274A1?style=for-the-badge)

## 💻 How to Run

To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/bulldozer-price-prediction.git](https://github.com/your-username/bulldozer-price-prediction.git)
    cd bulldozer-price-prediction
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(If no `requirements.txt` is provided, install: `pip install numpy pandas scikit-learn matplotlib seaborn jupyter`)*

3.  **Download the dataset:**
    Download the data from the [Kaggle competition page](https://www.kaggle.com/c/bluebook-for-bulldozers/data). Place the `TrainAndValid.csv` and `Test.csv` files inside a `data/bluebook-for-bulldozers/` directory within the project folder.

4.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open the main notebook file (`.ipynb`) and run the cells to see the complete analysis.
