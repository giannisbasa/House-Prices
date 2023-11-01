# LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns', None)
import os
import sys
import pathlib
import datetime
from dateutil.relativedelta import relativedelta
import itertools
from scipy.stats import chi2_contingency
from itertools import product
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_score
import xgboost as xg
from fast_ml.feature_selection import get_duplicate_features
import math
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn import set_config

# Specify the directory from where I will get the functions 
sys.path.insert(1, '/Users/macbookpro/Desktop/PYTHON/1. PROJECTS/1. House Prices/3. Coding')

from Functions import to_delete, to_keep, duplicate_features, distribution, box_plot, distplot, histplot, distribution_unique,\
calculate_fences, winzorning, impute_categorical_na, cardinality, datetime_calc, group_categorical_features, scale_condition

import warnings
warnings.filterwarnings("ignore")



# LOAD DATA
# Sample submission
df = pd.read_csv(r'/Users/macbookpro/Desktop/PYTHON/1. PROJECTS/1. House Prices/house-prices-advanced-regression-techniques/train.csv')

# Copy
data = df.copy()



# PIPELINE
## NUMERIC FEATURES
### Select only numeric variables 
number = data.select_dtypes(include=['number'])
num_var = [col for col in number if 'Year' not in col and 'Yr' not in col] # Numeric variables without datetime
num_df = data[num_var]

### Custom transformer for checking unique value distribution
class UniqueDistributionChecker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        col_list = []
        for col in X.columns:
            counts = X[col].value_counts(dropna=False, normalize=True)
            valids = counts[counts < 0.95].index
            if valids.any():  # check if the index is not empty
                col_list.append(col)
        return X[col_list]

### Custom transformer for calculating fences and handling outliers
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, target_columns=['Id', 'SalePrice'], iqr_multiplier=1.5):
        self.target_columns = target_columns
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fences_df = calculate_fences(X, self.target_columns, self.iqr_multiplier)
        # Apply low and high fence filtering to the data
        for index, row in fences_df.iterrows():
            col = row['Column']
            low_fence = row['Low Fence']
            high_fence = row['High Fence']

            X.loc[X[col] < low_fence, col] = low_fence
            X.loc[X[col] > high_fence, col] = high_fence

        return X

### Create a pipeline for numeric feature preprocessing
numeric_pipeline = pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=False)),  # Impute missing values with median
    ('unique_distribution_checker', UniqueDistributionChecker()),  # Check unique value distribution
    ('outlier_handler', OutlierHandler()),  # Calculate fences and handle outliers
], verbose=True)



## CATEGORICAL FEATURES
### Specify categorical variables
cat_df = data.select_dtypes(exclude=['number']) # Categorical dataset
cat_var = cat_df.columns # Categorical variables

### Custom imputation of categorical features
class CategoricalImputation(BaseEstimator, TransformerMixin):
    def __init__(self, columns, value = 'No Data'):
        self.columns = columns
        self.value = value

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].fillna(self.value)
        return X_copy

### Cardinality
class CardinalityReducer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = None, inplace = False):
        self.threshold = threshold
        self.inplace = inplace

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not self.inplace:
            X_copy = X.copy()  # Create a copy to preserve the original data
        else:
            X_copy = X  # Use the original DataFrame if inplace is True

        if self.threshold is None:
            # Calculate the threshold for high cardinality using the rule
            N = X_copy.shape[0]
            self.threshold = round(10 * math.sqrt(N))
           
        cardinality_columns = []
        for col in X_copy.columns:
            cardinality = len(X_copy[col].value_counts())
            if cardinality > self.threshold or cardinality == 1:
                cardinality_columns.append(col)

        X_result = X_copy.drop(cardinality_columns, axis=1)
        return X_result

### Create a pipeline for categorical feature preprocessing
categorical_pipeline = pipeline([
    ('imputer', CategoricalImputation(columns=cat_var, value='No Data')),  # Impute missing values with 'No Data'
    ('cardinality', CardinalityReducer(threshold=None, inplace=False)), # Check and reduce cardinality
    ('unique_distribution_checker', UniqueDistributionChecker()),  # Check unique value distribution
])



## DATETIME FEATURES
### Specify datetime variables
dt_var = [col for col in number if 'Year' in col or 'Yr' in col] # Datetime variables
dt_df = data[datetime]

### Create a pipeline for datetime features 
class DatetimeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, strategy = 'most_frequent'):
        self.columns = columns
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy == 'most_frequent':
            self.most_frequent_values = X[self.columns].mode().iloc[0]
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if self.strategy == 'most_frequent':
                X_copy[col].fillna(self.most_frequent_values[col], inplace = True)
        return X_copy

### Create new datetime columns from the difference of two dates
class DatetimeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2
        self.new_col_name =self.var1 + '_' + self.var2

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.new_col_name] = X_copy[self.var1] - X_copy[self.var2]
        return X_copy

### Create a list of datetime calculations to perform 
datetime_calculations = [
    {
        'var1': 'YrSold',
        'var2': 'YearBuilt',
        'new_col_name': 'YrSold_YearBuilt'
    },
     {
        'var1': 'YrSold',
        'var2': 'YearRemodAdd',
        'new_col_name': 'YrSold_YearRemodAdd'
    },
         {
        'var1': 'YrSold',
        'var2': 'GarageYrBlt',
        'new_col_name': 'YrSold_GarageYrBlt'
    }
]

### Create a pipeline for datetime feature preprocessing and calculations 
datetime_pipelines = []
for calculation in datetime_calculations:
    var1 = calculation['var1']
    var2 = calculation['var2']
    new_col_name = calculation['new_col_name']

    datetime_pipeline = pipeline([
        ('calculator', DatetimeCalculator(var1=var1, var2=var2, new_col_name=new_col_name)),
    ])
    datetime_pipelines.append((new_col_name, datetime_pipeline))

datetime_pipeline = pipeline([
    ('imputer', DatetimeImputer(columns=dt_var, strategy='most_frequent')), # Impute missing values with most frequent value
])



## FEATURE SCALING
### Use a custom function in the pipeline
categorical_group = ['LotShape', 'Alley', 'LandContour', 'Condition1', 'Exterior1st', 'Exterior2nd',\
                      'MasVnrType', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'SaleType']
groupin_transformer = FunctionTransformer(group_categorical_features)





## COMBINE ALL ABOVE
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_pipeline, num_var),
        ('categorical', categorical_pipeline, cat_var),
        ('datetime', datetime_pipeline, dt_var),
        ('group_categorical', groupin_transformer, categorical_group)
    ]
)


