#!/usr/bin/env python
# coding: utf-8

# # Titanic - Machine Learning from Disaster
# 
# This script presents a solution for the Kaggle competition "Titanic - Machine Learning from Disaster".
# It includes data exploration, preprocessing, feature engineering, model training, and prediction.

# ## Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ## Load Data
print("Loading data...")
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# ## Data Exploration
def explore_data(data):
    print("\nData Exploration:")
    print("Missing values:")
    print(data.isnull().sum())
    
    # Survival statistics (only for training data)
    if 'Survived' in data.columns:
        survival_rate = data['Survived'].mean() * 100
        print(f"\nSurvival rate: {survival_rate:.2f}%")
        
        # Gender survival
        gender_survival = data.groupby('Sex')['Survived'].mean() * 100
        print(f"Male survival rate: {gender_survival['male']:.2f}%")
        print(f"Female survival rate: {gender_survival['female']:.2f}%")
        
        # Class survival
        class_survival = data.groupby('Pclass')['Survived'].mean() * 100
        print(f"1st class survival rate: {class_survival[1]:.2f}%")
        print(f"2nd class survival rate: {class_survival[2]:.2f}%")
        print(f"3rd class survival rate: {class_survival[3]:.2f}%")

# Run exploration
explore_data(train_data)

# ## Feature Engineering
print("\nPerforming feature engineering...")

# Save the target variable
train_labels = train_data['Survived']

# Add a temporary column to identify train and test data
train_data['IsTrainSet'] = True
test_data['IsTrainSet'] = False

# Combine the datasets
combined_data = pd.concat([train_data.drop('Survived', axis=1), test_data], axis=0).reset_index(drop=True)
print(f"Combined data shape: {combined_data.shape}")

# Extract titles from names
combined_data['Title'] = combined_data['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

# Map titles to categories
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Capt': 'Officer',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Sir': 'Royalty',
    'the Countess': 'Royalty',
    'Dona': 'Royalty',
    'Lady': 'Royalty',
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mme': 'Mrs'
}

combined_data['Title'] = combined_data['Title'].map(title_mapping)
combined_data['Title'] = combined_data['Title'].fillna('Other')

# Create family features
combined_data['FamilySize'] = combined_data['SibSp'] + combined_data['Parch'] + 1
combined_data['IsAlone'] = (combined_data['FamilySize'] == 1).astype(int)

# Create a more detailed family type
combined_data['FamilyType'] = 'Small'
combined_data.loc[combined_data['FamilySize'] == 1, 'FamilyType'] = 'Single'
combined_data.loc[combined_data['FamilySize'] > 4, 'FamilyType'] = 'Large'

# Extract cabin deck
combined_data['Deck'] = combined_data['Cabin'].str.slice(0, 1)
combined_data['Deck'] = combined_data['Deck'].fillna('U')  # U for Unknown

# Fill missing ages based on title and class
def fill_age(data):
    # Group by Title and Pclass
    age_by_title_class = data.groupby(['Title', 'Pclass'])['Age'].median()
    
    # Fill missing values with the group median
    for (title, pclass), age in age_by_title_class.items():
        data.loc[(data['Age'].isnull()) & 
                 (data['Title'] == title) & 
                 (data['Pclass'] == pclass), 'Age'] = age
    
    # For any remaining nulls, use the global median
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    return data

combined_data = fill_age(combined_data)
print(f"Missing ages after filling: {combined_data['Age'].isnull().sum()}")

# Create age groups
combined_data['AgeGroup'] = pd.cut(combined_data['Age'], 
                              bins=[0, 12, 18, 30, 50, 80],
                              labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])

# Fill missing Embarked values with the most common
combined_data['Embarked'] = combined_data['Embarked'].fillna(combined_data['Embarked'].mode()[0])

# Fill missing Fare with median by Pclass
combined_data['Fare'] = combined_data.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

# Create fare groups
combined_data['FareGroup'] = pd.qcut(combined_data['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

# ## Feature Selection

# Select features for modeling
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'FamilySize', 'Title', 'IsAlone', 'FamilyType', 'Deck', 'AgeGroup', 'FareGroup']

# Split combined data back into train and test
train_processed = combined_data[combined_data['IsTrainSet'] == True][features].reset_index(drop=True)
test_processed = combined_data[combined_data['IsTrainSet'] == False][features].reset_index(drop=True)

print(f"Processed training data shape: {train_processed.shape}")
print(f"Processed test data shape: {test_processed.shape}")

# ## Model Training

print("\nTraining models...")

# Define numeric and categorical features
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'FamilyType', 'Deck', 'AgeGroup', 'FareGroup']

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create models to try
models = {
    'LogisticRegression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    
    'RandomForest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    
    'GradientBoosting': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    
    'SVC': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))
    ])
}

# Define cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate each model with cross-validation
for name, model in models.items():
    scores = cross_val_score(model, train_processed, train_labels, cv=cv, scoring='accuracy')
    print(f"{name} - Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Hyperparameter tuning for RandomForest
print("\nTuning RandomForest hyperparameters...")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_processed, train_labels)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Get the best model
best_rf_model = grid_search.best_estimator_

# Hyperparameter tuning for Gradient Boosting
print("\nTuning Gradient Boosting hyperparameters...")
gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 5],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(gb_pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_processed, train_labels)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Get the best model
best_gb_model = grid_search.best_estimator_

# Create ensemble model
print("\nCreating ensemble model...")
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', best_rf_model),
        ('gb', best_gb_model),
        ('svc', models['SVC'])
    ],
    voting='soft'
)

# Train ensemble
ensemble_model.fit(train_processed, train_labels)

# Evaluate ensemble model
ensemble_score = cross_val_score(ensemble_model, train_processed, train_labels, cv=cv, scoring='accuracy')
print(f"Ensemble model - Cross-validation accuracy: {ensemble_score.mean():.4f} ± {ensemble_score.std():.4f}")

# ## Make Predictions and Create Submission File
print("\nGenerating predictions...")
test_predictions = ensemble_model.predict(test_processed)

# Create submission dataframe
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})

# Save submission file
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)

print(f"Submission file created at {submission_path}")
print("Preview of submission file:")
print(submission.head())

print("\nFeature importance:")
# Get feature names
feature_names = (numeric_features + 
                 list(best_rf_model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features)))

# Print top 10 important features
rf_importances = best_rf_model.named_steps['classifier'].feature_importances_
indices = np.argsort(rf_importances)[::-1]
print("Top 10 most important features:")
for i in range(min(10, len(indices))):
    print(f"{feature_names[indices[i]]}: {rf_importances[indices[i]]:.4f}")

print("\nProcess completed successfully!") 