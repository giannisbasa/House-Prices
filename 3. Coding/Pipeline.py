try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn import set_config
import unittest

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
data = data.loc[:49, :].copy() # Create a subset of the dataset for unit testing
print(data.shape)


# PIPELINE
# 1. NUMERIC FEATURES
# STEP 1: DEFINE FEATURES TO USE
number = data.select_dtypes(include=['number'])
num_var = [col for col in number if 'Year' not in col and 'Yr' not in col] # Numeric variables without datetime
num_df = data[num_var]


# STEP 2: CREATE CUSTOM TRANSFORMATIONS  
class DataFrameImputer(TransformerMixin):
    def __init__(self, strategy='median', add_indicator=False):
        self.strategy = strategy
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy, add_indicator=self.add_indicator)
        self.imputer.fit(X)
        return self

    def transform(self, X):
        imputed_data = self.imputer.transform(X)
        return pd.DataFrame(imputed_data, columns=X.columns)


# Custom transformer for checking unique value distribution
class UniqueDistributionChecker(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
            if isinstance(X, pd.DataFrame):
                col_list = []
                for col in X.columns:
                    counts = X[col].value_counts(dropna=False, normalize=True)
                    valids = counts[counts < 0.95].index
                    if valids.any():  # check if the index is not empty
                        col_list.append(col)
                return X[col_list]
            elif isinstance(X, np.ndarray):
                # Assume X is a numpy array
                # Implement your logic here to filter columns in numpy array if needed
                return X  # For now, simply return X as-is
            else:
                raise ValueError("Input must be a DataFrame or numpy array")
            
# Custom transformer for calculating fences and handling outliers
class OutlierHandler(BaseEstimator, TransformerMixin): 
    def __init__(self, target_columns=['Id', 'SalePrice'], iqr_multiplier=1.5):
        self.target_columns = target_columns
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        self.column_info = {}  # Store fence information for each column
        for col in X.select_dtypes(include='number').columns:
        # for col in X.columns:
            if col not in self.target_columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                low_fence = q1 - self.iqr_multiplier * iqr
                high_fence = q3 + self.iqr_multiplier * iqr
                self.column_info[col] = {'Low Fence': low_fence, 'High Fence': high_fence}
        return self

    def transform(self, X):
        for col, info in self.column_info.items():
            low_fence = info['Low Fence']
            high_fence = info['High Fence']
            X.loc[X[col] < low_fence, col] = low_fence
            X.loc[X[col] > high_fence, col] = high_fence

        return X


# STEP 3: DEFINE STEPS 
numeric_steps = [
    ('imputer', DataFrameImputer(strategy='median', add_indicator=False)),  # Impute missing values with median)
    ('unique_distribution_checker', UniqueDistributionChecker()),  # Check unique value distribution
    ('outlier_handler', OutlierHandler()),  # Calculate fences and handle outliers
]


# STEP 4: CREATE A PIPELINE (Numeric Transformer)
numeric_pipeline = Pipeline(numeric_steps, verbose=True)



# 2. CATEGORICAL FEATURES
# STEP 1: DEFINE FEATURES TO USE
cat_df = data.select_dtypes(exclude=['number']) # Categorical dataset
cat_var = cat_df.columns # Categorical variables


# STEP 2: CREATE CUSTOM TRANSFORMATIONS  
# Categorical imputation
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

# Cardinality
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


# STEP 3: DEFINE STEPS 
categorical_steps = [
    ('imputer', CategoricalImputation(columns=cat_var, value='No Data')),  # Impute missing values with 'No Data'
    ('cardinality', CardinalityReducer(threshold=None, inplace=False)), # Check and reduce cardinality
    ('unique_distribution_checker', UniqueDistributionChecker()),  # Check unique value distribution
]


# STEP 4: CREATE A PIPELINE (Categorical Transformer)
categorical_pipeline = Pipeline(categorical_steps)



# 3. DATETIME FEATURES
# STEP 1: DEFINE FEATURES TO USE
dt_var = [col for col in number if 'Year' in col or 'Yr' in col] # Datetime variables
dt_df = data[dt_var]


# STEP 2: CREATE CUSTOM TRANSFORMATIONS 
# 3.1 Create a pipeline for datetime features 
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

# 3.2 Create new datetime columns from the difference of two dates
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

# Create a list of datetime calculations to perform 
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


# STEP 3-4: DEFINE STEPS & CREATE A PIPELINE (Datetime Transformer)
datetime_pipelines = []

for calculation in datetime_calculations:
    var1 = calculation['var1']
    var2 = calculation['var2']
    new_col_name = calculation['new_col_name']

    datetime_pipeline = Pipeline([
        ('calculator', DatetimeCalculator(var1=var1, var2=var2)),
    ])
    datetime_pipelines.append((new_col_name, datetime_pipeline))

final_datetime_pipeline = Pipeline([
    ('imputer', DatetimeImputer(columns=dt_var, strategy='most_frequent')), # Impute missing values with most frequent value
] + datetime_pipelines)



# 4. FEATURE SCALING
# STEP 1: DEFINE FEATURES TO USE
# 4.1 Custom scaling Categorical features
categorical_group = ['LotShape', 'Alley', 'LandContour', 'Condition1', 'Exterior1st', 'Exterior2nd',\
                      'MasVnrType', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'SaleType']


# STEP 2: CREATE CUSTOM TRANSFORMATIONS
grouping_transformer = FunctionTransformer(group_categorical_features)

# 4.2 Custom scaling Numeric features
# Define condition mappings for each column
condition_mappings = {
    "GarageCond": {
        "No Garage": 1,
        "Po": 2,
        "Fa": 3,
        'TA': 4,
        'Gd': 5,
        'Ex': 6
    },
    "GarageQual": {
       "No Garage": 1,
        "Po": 2,
        "Fa": 3,
        'TA': 4,
        'Gd': 5,
        'Ex': 6
    },
    "KitchenQual": {
        "Fa": 1,
        'TA': 2,
        'Gd': 3,
        'Ex': 4
    },
    "ExterQual": {
        "Fa": 1,
        'TA': 2,
        'Gd': 3,
        'Ex': 4
    },
    "ExterCond": {
        "Po": 1,
        "Fa": 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5
    },
    "HeatingQC": {
        "Po": 1,
        "Fa": 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5
    },
    "BsmtCond": {
        "No Basement": 1,
        "Po": 2,
        "Fa": 3,
        'TA': 4,
        'Gd': 5
    },
    "BsmtQual": {
        "No Basement": 1,
        "Fa": 2,
        'TA': 3,
        'Gd': 4,
        'Ex': 5
    },
    "FireplaceQu": {
        "No Fireplace": 1,
        "Po": 2,
        "Fa": 3,
        'TA': 4,
        'Gd': 5,
        'Ex': 6
    }
}

# Specify the columns to scale
columns_to_scale = list(condition_mappings.keys())

class ConditionScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, mapping):
        self.columns = columns
        self.mapping = mapping

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in self.mapping:
                X_copy[col] = X_copy[col].map(self.mapping).fillna(X_copy[col])
        return X_copy
    
condition_scaler = ConditionScaler(columns=categorical_group, mapping=condition_mappings)

# 4.3 Numeric features | Standar Scaler
class NumericScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, scaler):
        self.columns = columns
        self.scaler = scaler

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns] = self.scaler.transform(X[self.columns])
        return X_copy
    
num_columns_to_scale = data.select_dtypes(include=['number']).columns
numeric_scaling_transformer = NumericScaler(columns=num_columns_to_scale, scaler=StandardScaler())

# 4.4 Categorical features | Onehotencoding
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, encoder):
        self.columns = columns
        self.encoder = encoder

    def fit(self, X, y=None):
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        encoded_columns = self.encoder.transform(X[self.columns])
        feature_names = self.encoder.get_feature_names_out(input_features=self.columns)
        X_encoded = pd.DataFrame(encoded_columns, columns=feature_names, index=X_copy.index)
        X_copy = pd.concat([X_copy, X_encoded], axis=1)
        X_copy = X_copy.drop(self.columns, axis=1)
        return X_copy

# Modify the categorical_encoding_transformer as follows
categorical_columns_to_encode = data.select_dtypes(include=['object']).columns
categorical_encoding_transformer = CategoricalEncoder(columns=categorical_columns_to_encode, encoder=OneHotEncoder(sparse=False, drop='first'))



## 5. COMBINE ALL ABOVE
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_pipeline, num_var),
        ('categorical', categorical_pipeline, cat_var),
        ('datetime', final_datetime_pipeline, dt_var),
        ('group_categorical', grouping_transformer, categorical_group),
        ('condition_scaling', condition_scaler, columns_to_scale), 
        ('numeric_scaling', numeric_scaling_transformer, num_columns_to_scale),
        ('categorical_encoding', categorical_encoding_transformer, categorical_columns_to_encode)
    ]
)



# CREATE FINAL PIPELINE
final_steps = [
    (preprocessor)
]

final_pipeline = Pipeline(final_steps)

preprocessor.fit(data)

preprocessed_data = preprocessor.transform(pd.DataFrame(data))
preprocessed_data
