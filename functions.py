#functions to be used in the data preparation process

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def market_columns(df):
    """Function that maps multiple entries in a column into individual columns
    and assings a score of 1 or 0 if the entry is question is within a given row.
    """
    categories = []
    for category in list(df['Market Category'].unique()):
        categories += category.split(',')

    unique = set(categories)
    for col in unique:
        df[col] = df['Market Category'].apply(lambda x: 1 if col in x.split(',') else 0)
    df.drop('Market Category', axis=1, inplace=True)
    return df

def onehotencode(X):
    """
    One hot encode the categorical variables in the dataframe to convert them to numerical variables.
    """
    X_obj = X[[col for col,dtype in list(zip(X.columns, X.dtypes)) 
                          if dtype == np.dtype('O')]]
    
    X_nonobj = X[[col for col,dtype in list(zip(X.columns, X.dtypes)) 
                          if dtype != np.dtype('O')]]
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_obj_ohe = ohe.fit_transform(X_obj)
    
    X_nonobj_df = pd.DataFrame(X_nonobj).reset_index(drop=True) 
    X_obj_ohe_df = pd.DataFrame(X_obj_ohe.todense(), columns=ohe.get_feature_names()).reset_index(drop=True)
    
    X_all = pd.concat([X_nonobj_df, X_obj_ohe_df], axis=1)
   
    return X_all