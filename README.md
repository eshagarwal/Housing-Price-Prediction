# California Housing Price Prediction

This project analyzes the California Housing dataset to build a machine learning model that predicts housing prices based on various features like location, demographics, and housing characteristics.

## Project Overview

The notebook `housing_price_prediction.ipynb` contains a complete data science pipeline that:

- Loads and explores the California Housing dataset  
- Preprocesses the data with various techniques  
- Engineers meaningful features  
- Builds and compares multiple machine learning models  
- Fine-tunes the best model using hyperparameter optimization  

## Key Features

- **Data Preprocessing**: Handles missing values, outlier detection, and categorical data encoding  
- **Feature Engineering**: Creates informative features like rooms per house, bedrooms ratio, and people per house  
- **Advanced Transformation Pipelines**: Uses scikit-learn's `Pipeline` and `ColumnTransformer`  
- **Custom Transformers**: Implements custom transformers for specialized feature transformations  
- **Model Comparison**: Evaluates Linear Regression, Decision Trees, and Random Forest models  
- **Hyperparameter Tuning**: Uses `GridSearchCV` and `RandomizedSearchCV` for optimization  
- **Model Evaluation**: Employs cross-validation and confidence intervals for robust performance assessment  

## Models Implemented

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  

## Dataset

The dataset is stored in `housing.csv` and contains information about California housing districts including:

- Geographic location (longitude, latitude)  
- Housing attributes (rooms, bedrooms, age)  
- Demographics (population, households, income)  
- **Target variable**: median house value  

## Results

The **Random Forest Regressor** with optimized hyperparameters performed best. Feature importance analysis revealed **median income** and **geographical location** as the strongest predictors of housing prices.
