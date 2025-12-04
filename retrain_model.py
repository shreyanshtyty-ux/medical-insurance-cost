#!/usr/bin/env python3
"""
Script to retrain the medical insurance cost model with current scikit-learn version
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Load the training data
print("Loading training data...")
train_data = pd.read_csv('Train_Data.csv')

# Prepare features
print("Preparing features...")
X = train_data.drop('charges', axis=1)
y = train_data['charges']

# Encode categorical variables
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

X['sex'] = le_sex.fit_transform(X['sex'])
X['smoker'] = le_smoker.fit_transform(X['smoker'])
X['region'] = le_region.fit_transform(X['region'])

# Create dummy variables to match the app.py format
X_encoded = pd.get_dummies(train_data[['sex', 'smoker', 'region']], drop_first=True)
X_final = pd.concat([
    train_data[['age', 'bmi', 'children']],
    X_encoded
], axis=1)

# Train the model
print("Training Decision Tree Regressor...")
model = DecisionTreeRegressor(random_state=42)
model.fit(X_final, y)

# Save the model
print("Saving model...")
with open('MedicalInsuranceCost.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model retrained and saved successfully!")
print(f"Model score: {model.score(X_final, y):.4f}")
