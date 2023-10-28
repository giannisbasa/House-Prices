## LIBRARIES ##
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns', None)
import os
import pathlib
import datetime
from dateutil.relativedelta import relativedelta
import itertools
from scipy.stats import chi2_contingency
from itertools import product
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler
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

import warnings
warnings.filterwarnings("ignore")


# Sample submission
df = pd.read_csv(r'/Users/macbookpro/Desktop/PYTHON/1. PROJECTS/1. House Prices/house-prices-advanced-regression-techniques/train.csv')

# Copy
data = df.copy()

# Select only numeric variables 
number = data.select_dtypes(include=['number'])

datetime = [col for col in number if 'Year' in col or 'Yr' in col] # Datetime variables
dt_df = data[datetime]

num_var = [col for col in number if 'Year' not in col and 'Yr' not in col] # Numeric variables without datetime
num_df = data[num_var]

cat_df = data.select_dtypes(exclude=['number']) # Categorical variables







# GENERAL FUNCTIONS
## 1.1 Function for features to delete
def to_delete(df, name):
    path = r'/Users/macbookpro/Desktop/PYTHON/1. PROJECTS/1. House Prices/2. Delete/'
    # name = input('Name of the file: ')
    if type(df) == list:
        df1 = pd.DataFrame(df, columns = ['Columns'])
        df1.to_csv(path + "/" + name, header=None, index=None, sep=",", mode="w")
    else:
        df.to_csv(path + "/" + name, header=None, index=None, sep=",", mode="w")


## 1.2 Function for features to keep
def to_keep(df, name):
    path = r'/Users/macbookpro/Desktop/PYTHON/1. PROJECTS/1. House Prices/1. Keep/'
    # name = input('Name of the file: ')
    if type(df) == list:
        df1 = pd.DataFrame(df, columns = ['Columns'])
        df1.to_csv(path + "/" + name, header=None, index=None, sep=",", mode="w")
    else:
        df.to_csv(path + "/" + name, header=None, index=None, sep=",", mode="w")


## 1.3 Function for indetify if two columns have the same values
def duplicate_features(df, var1, var2):
    df_dupl = df[[var1, var2]].apply(pd.Series.value_counts)
    df_dupl['Equal'] = np.where(df_dupl[var1] == df_dupl[var2], 1,0)
    print('Columns have the same values: ', df_dupl['Equal'].sum() == df_dupl.shape[0])
    return df_dupl.head(2)


## 1.4 compute the vif for all given features
def compute_vif(considered_features):
    
    X = train[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif


## 1.5 Calculate the distribution of a value
def distribution(df, x):
    return df[x].value_counts(normalize=True).round(2).mul(100)



# GRAPHS 
## 2.1 Boxplot
def box_plot(df, x):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(15,5))
    plt.title('Boxplot');
    sns.boxplot(df[x]);


## 2.2 Distplot
def distplot(df, x):  
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(15,5))
    sns.distplot(df[x])

    print('Skewness of the {} is : {}'.format(x, df[x].skew()))

    plt.axvline(x = df[x].mean(), linewidth = 3, color='g', label = 'mean', alpha=.5)
    plt.axvline(x = df[x].median(), linewidth = 3, color='r', label = 'median', alpha=.5)

    plt.legend()
    plt.show()

## 2.3 Histplot
def histplot(df, x):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(15,5))
    plt.title('Histplot');
    sns.histplot(df[x]);



# NUMERIC FEATURES
## Distribution of Unique Values
def distribution_unique(df):
    col_list = []
    for col in df.columns:
        counts = df[col].value_counts(dropna = False, normalize = True)
        valids = counts[counts>.95].index
        if valids.any(): # check if the index is not empty
            col_list.append(col)
    
    print('Number of features is: ', len(col_list))
    print('Features:', col_list)


## OUTLIERS
### Calculate high and low fences 
def calculate_fences(df, target_columns=['Id','SalePrice'], iqr_multiplier=1.5):
    """
    Calculate low and high fences for numeric columns in a DataFrame.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - target_columns: list of str, optional
        Names of columns to exclude from calculations. Default is None.
    - iqr_multiplier: float, optional
        Multiplier for the IQR to determine fences. Default is 1.5.

    Returns:
    - DataFrame containing columns: 'Column', 'Low Fence', 'High Fence'.
    """
    if target_columns is None:
        target_columns = []  # Default to an empty list

    col_list = []
    low_ls = []
    high_ls = []

    for col in df.select_dtypes(include='number').columns:
        if col not in target_columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            low_fence = q1 - iqr_multiplier * iqr
            high_fence = q3 + iqr_multiplier * iqr

            col_list.append(col)
            low_ls.append(low_fence)
            high_ls.append(high_fence)

    fences_df = pd.DataFrame({'Column': col_list, 'Low Fence': low_ls, 'High Fence': high_ls})
    return fences_df


### Create a new dataframe having replace extreme values
def winzorning(df, fences_df, inplace=False):
    """
    Replace outliers in a DataFrame with the low and high fences from fences_df.

    Parameters:
    - df: DataFrame
        The input DataFrame containing potential outliers.
    - fences_df: DataFrame
        The DataFrame containing columns 'Column', 'Low Fence', and 'High Fence'.
    - inplace: bool, optional
        If True, replace outliers in the original DataFrame in-place. Default is False.

    Returns:
    - DataFrame with outliers replaced with fence values.
    """
    if not inplace:
        df = df.copy()  # Create a copy to preserve the original data

    for index, row in fences_df.iterrows():
        col = row['Column']
        low_fence = row['Low Fence']
        high_fence = row['High Fence']

        # Replace values below the low fence with the low fence value
        df.loc[df[col] < low_fence, col] = low_fence

        # Replace values above the high fence with the high fence value
        df.loc[df[col] > high_fence, col] = high_fence

    return df



# CATEGORICAL FEATURES
## Impute missing values
def impute_categorical_na(df, columns, value = 'No Data'):
    for col in columns:
        df[col] = df[col].fillna(value)

    return df


## Cardinality
def cardinality(df, threshold=None, inplace = False):
    """
    Identify columns with high cardinality and low cardinality.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - threshold: int, optional
        The cardinality threshold. If None, the threshold is calculated based on the rule.
        Default is None.

    Returns:
    - DataFrame containing columns 'Column' and 'Cardinality'.
    """

    if not inplace:
        df_copy = df.copy()  # Create a copy to preserve the original data
    else:
        df_copy = df  # Use the original DataFrame if inplace is True

    if threshold is None:
        # Calculate the threshold for high cardinality using the rule
        N = df_copy.shape[0]
        threshold = round(10 * math.sqrt(N))

    sthles = []

    for col in df_copy:
        card = len(df_copy[col].value_counts())
        if card == 1 or card > threshold:
            sthles.append(col)

    result_df = df_copy[[col for col in df_copy if col not in sthles]]
    return result_df



# DATETIME
## Create new datetime columns from the difference of two dates
def datetime_calc(df, var1, var2):
    new_col = var1 + '_' + var2 # Construct the new column name
    df[new_col] = df[var1] - df[var2] # Calculate the date difference and store it in the new column

    return df






