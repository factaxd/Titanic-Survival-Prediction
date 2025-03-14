# Titanic - Machine Learning from Disaster

This repository contains a comprehensive solution for the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) Kaggle competition. The goal is to predict which passengers survived the Titanic shipwreck.

## Solution Overview

The solution includes:

1. **Data Exploration** - Analyzing the training data to understand patterns
2. **Feature Engineering** - Creating new features to improve model performance
3. **Model Training** - Training multiple ML models and creating an ensemble
4. **Prediction** - Generating predictions for submission

## Key Features

- Comprehensive data preprocessing
- Feature extraction from passenger names (titles)
- Family size and deck information extraction
- Missing value imputation based on passenger characteristics
- Model ensemble combining Random Forest, Gradient Boosting, and SVM

## Files

- `titanic_solution.py` - Complete Python script solution
- `titanic_solution.ipynb` - Jupyter notebook version with visualizations
- `requirements.txt` - Required Python packages
- `submission.csv` - Generated predictions for competition submission

## Dataset

The dataset should be placed in the `data/` directory with the following files:
- `train.csv` - Training data
- `test.csv` - Test data for predictions
- `gender_submission.csv` - Example submission file

## Getting Started

1. **Clone this repository**

2. **Install required packages**
   ```
   pip install -r requirements.txt
   ```

3. **Run the solution**
   
   For Python script:
   ```
   python titanic_solution.py
   ```
   
   For Jupyter notebook:
   ```
   jupyter notebook titanic_solution.ipynb
   ```

4. **Submit predictions**
   
   The `submission.csv` file will be generated, which can be submitted to Kaggle.

## Model Performance

The solution achieves approximately 80-82% accuracy on cross-validation. 

**Kaggle Competition Score: 0.77990**

This score places the solution in a competitive position on the Kaggle leaderboard. The score was achieved using the ensemble approach of combining Random Forest, Gradient Boosting, and SVM classifiers.

## Key Insights

- Gender was a crucial factor in survival (females had much higher survival rates)
- Passenger class strongly correlated with survival (1st class passengers had better chances)
- Age played an important role (children were prioritized)
- Family size affected survival chances

## Feature Importance

The top features that contributed most to prediction accuracy were:
1. Sex (gender)
2. Title extracted from name
3. Fare
4. Age
5. Passenger class

## Model Development Process

The solution followed a systematic approach:
1. Initial data cleaning and exploration
2. Feature engineering to create new predictive variables
3. Testing multiple models independently
4. Hyperparameter tuning for best performing models
5. Creating an ensemble of the top models
6. Final prediction on the test dataset

## Further Improvements

Potential ways to improve the model:
- More advanced feature engineering
- Additional models in the ensemble
- More extensive hyperparameter tuning
- Neural network implementation
- Additional external data sources
- Advanced imputation techniques for missing values 