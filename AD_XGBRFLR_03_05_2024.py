#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:30:00 2024


@author: Ajay
"""

#%% Loading appropriate libraries ===============================================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics, linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE

#%% Loading data ===============================================================
# Provide the file path
file_path = 'C:/Users/ADhungana/Desktop/DIGIAG/TG_ALLDATA_05_17_2021.csv'

# Read the CSV file into a DataFrame
Ex1 = pd.read_csv(file_path)


#%% Removing entries with missing values ==============================
Ex1_cleaned = Ex1.dropna()

#%% Remove outlier yields (top 5% and bottom 5%)
top_percentile = Ex1_cleaned['YIELD'].quantile(0.95)
bottom_percentile = Ex1_cleaned['YIELD'].quantile(0.05)
Ex1_cleaned = Ex1_cleaned[(Ex1_cleaned['YIELD'] >= bottom_percentile) & (Ex1_cleaned['YIELD'] <= top_percentile)]

#%% Description of the dataset and the columns, re-arranging so that Yield is first
cols = list(Ex1_cleaned)
cols.insert(0, cols.pop(cols.index('YIELD')))
Ex1_cleaned = Ex1_cleaned.loc[:,cols]

#%% Converting character to numeric dummy variables
Ex1feat = pd.get_dummies(Ex1_cleaned)
print('The shape of the Dataset with Dummy Variables is :', Ex1feat.shape)

#%% Training and test separations ===============================================
X = Ex1feat.iloc[:,1:].values
y = Ex1feat.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Data normalization of the training and test sets
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Setting up hyperparameters for the random forest by fitting  the dataset
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30],
}

grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=0), 
                              param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

print("Best parameters for Random Forest: ", grid_search_rf.best_params_)
print("Best RMSE score for Random Forest: ", np.sqrt(-grid_search_rf.best_score_))

regressor = grid_search_rf.best_estimator_

#%% Setting up XGBoost and fitting the dataset
# Setting up parameter grid
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
}

# Create an instance of XGBRegressor
xgb_regressor = xgb.XGBRegressor(random_state=0)

# Initialize GridSearchCV with the XGBRegressor instance and parameter grid
grid_search_xgb = GridSearchCV(estimator=xgb_regressor, 
                               param_grid=param_grid_xgb, cv=5, 
                               scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the GridSearchCV object to your data
grid_search_xgb.fit(X_train, y_train)

# Retrieve the best parameters
best_params_xgb = grid_search_xgb.best_params_
print(best_params_xgb)

#%% Training the Random Forest model
regressor.fit(X_train, y_train)  

y_pred_rf = regressor.predict(X_test)  

print('Mean Absolute Error for Random Forests:', metrics.mean_absolute_error(y_test, y_pred_rf))  
print('Mean Squared Error for Random Forests:', metrics.mean_squared_error(y_test, y_pred_rf))  
print('Root Mean Squared Error for Random Forests:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))  

#%% Creating linear regression object and computing metrics
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred_lr = regr.predict(X_test)

print('Mean Absolute Error for Linear Regression:', metrics.mean_absolute_error(y_test, y_pred_lr))  
print('Mean Squared Error for Linear Regression:', metrics.mean_squared_error(y_test, y_pred_lr))  
print('Root Mean Squared Error for Linear Regression:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))  

#%% Setting up XGBoost and fitting the dataset
xgb_regressor.fit(X_train, y_train)

y_pred_xgb = xgb_regressor.predict(X_test)

print('Mean Absolute Error for XGBoost:', metrics.mean_absolute_error(y_test, y_pred_xgb))
print('Mean Squared Error for XGBoost:', metrics.mean_squared_error(y_test, y_pred_xgb))
print('Root Mean Squared Error for XGBoost:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb)))

#%% ==== Create empty lists to store errors for 3 models(initialization)
rf_errors = []
regr_errors = []
xgb_errors = []

#%% ==== Create 100 repetitions
for i in range(100):
    # Simple train test split with 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    # Standardize input variables
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  

    # Random Forest 
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)  
    regressor.fit(X_train, y_train)  
    y_pred_rf = regressor.predict(X_test)  
    # Calculate MSE for Random Forest
    rf_errors.append(metrics.mean_squared_error(y_test, y_pred_rf))
    
    # Linear Regression 
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred_lr = regr.predict(X_test)
    # Calculate MSE for Linear Regression
    regr_errors.append(metrics.mean_squared_error(y_test, y_pred_lr))
    
    # XGBoost
    xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
    xgb_regressor.fit(X_train, y_train)
    y_pred_xgb = xgb_regressor.predict(X_test)
    # Calculate MSE for XGBoost
    xgb_errors.append(metrics.mean_squared_error(y_test, y_pred_xgb))

#%% ==== Combine three error arrays, clean up and select 100 repetitions at random, save as csv
Result_errors1 = pd.DataFrame({'Random_Forest': rf_errors, "Linear_Regression": regr_errors, "XGBoost": xgb_errors})
Result_errors2 = Result_errors1[Result_errors1["Linear_Regression"] < 100]
Result_errors3 = Result_errors2.sample(n=100, random_state=0)  # Selecting 100 random rows
Result_errors3.to_csv('//Users//admin//Desktop//Dig_Ag_Assignment//Data//Errors.csv')

#%% ==== Plotting Side by Side the RMSE
fig, ax = plt.subplots()
# Build a box plot
ax.boxplot(Result_errors3)
# Title and axis labels
ax.set_title('Side by Side Boxplot of RMSE for different Models')
ax.set_xlabel('Predictive Models')
ax.set_ylabel('Root Mean Square Errors')
xticklabels=['Random_Forest', 'Linear_Regression', 'XGBoost']
ax.set_xticklabels(xticklabels)
# Add horizontal grid lines
ax.yaxis.grid(True)
# Show the plot
plt.savefig('//Users//admin//Desktop//Dig_Ag_Assignment//Side_by_Side.png')

#%% ==== Describe Errors and save a csv file
Res = Result_errors3.describe()
Res.to_csv('//Users//admin//Desktop//Dig_Ag_Assignment//Data//Error_description.csv')

#%% ==== Plotting Side by Side the RMSE for Random Forest and Linear Regression
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

# Random Forest
axes[0].boxplot(Result_errors3['Random_Forest'])
axes[0].set_title('RMSE for Random Forest')

# Linear Regression
axes[1].boxplot(Result_errors3['Linear_Regression'])
axes[1].set_title('RMSE for Linear Regression')

plt.tight_layout()
plt.savefig('//Users//admin//Desktop//Dig_Ag_Assignment//RF_vs_LR_Side_by_Side.png')

#%% ==== Plotting Side by Side the RMSE for Random Forest and Linear Regression with XGBoost
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

# Random Forest with XGBoost
axes[0].boxplot(Result_errors3['XGBoost_on_RF'])
axes[0].set_title('RMSE for Random Forest with XGBoost')

# Linear Regression with XGBoost
axes[1].boxplot(Result_errors3['XGBoost_on_LR'])
axes[1].set_title('RMSE for Linear Regression with XGBoost')

plt.tight_layout()
plt.savefig('//Users//admin//Desktop//Dig_Ag_Assignment//XGB_on_RF_vs_LR_Side_by_Side.png')

#%% Computes best parameters and the corresponding regressor
best_params_rf = grid_search_rf.best_params_
df_rf = pd.DataFrame([best_params_rf])
rf_best_param_path = "//path//to//save//rf_best_param.csv"
df_rf.to_csv(rf_best_param_path, index=False)

best_params_xgb = grid_search_xgb.best_params_
df_xgb = pd.DataFrame([best_params_xgb])
xgb_best_param_path = "//path//to//save//xgb_best_param.csv"
df_xgb.to_csv(xgb_best_param_path, index=False)
