# Car Price Prediction using Machine Learning

## Overview
This project predicts the selling price of used cars based on various features such as present price, year of manufacture, fuel type, transmission type, kilometers driven, and ownership details.

The model was developed as part of the CodeAlpha Data Science Internship and uses a Random Forest Regressor for accurate price prediction.

## Features
- Data Cleaning and Preprocessing
- Feature Engineering (Car Age Calculation)
- Categorical Data Encoding
- Random Forest Regression Model
- Model Evaluation using MAE, MSE, RMSE, and R² Score
- Feature Importance Analysis
- Data Visualization

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Model Performance

| Metric | Value |
|----------|----------|
| MAE | 0.6191 |
| MSE | 0.8925 |
| RMSE | 0.9447 |
| R² Score | 0.9613 |

## Key Insights
- Present Price is the most influential factor affecting selling price.
- Vehicle age significantly impacts resale value.
- Cars with lower mileage generally have higher prices.
- Transmission type influences market value.

## Project Structure

CodeAlpha_CarPricePrediction/
│
├── car_price_prediction.py
├── car data.csv
├── README.md
└── screenshots/

## How to Run

1. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
