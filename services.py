import requests
import json
from logging_decorator import *
import pandas as pd
from scipy import stats
import numpy as np
from termcolor import colored as cl # text customization

class Services:
    def __init__(self):
        return

    # can potentially delete
    # @log
    # def add_dummies(self,df):
    #     categorical_cols = ["foreclosure","price_reduced","new_construction","new_listing","subdivision","postal_code","city"]
    #     df[categorical_cols] = df[categorical_cols].astype('category')
    #     for category in categorical_cols:
    #         df[category] = df[category].cat.codes
    #     return df

    # Can potentially delete
    #TODO number of garages, number of stories have data leakage.
    # @log
    # def clean(self,df):
    #     df = df[df['price'] < 1000000]
    #     df = df[df['house_type'] == "single_family"]
    #     df.dropna(subset=['sqft'], inplace=True)
    #     df['subdivision'].fillna(False, inplace=True)
    #     df['foreclosure'].fillna(False, inplace=True)
    #     df['price_reduced'].fillna(False, inplace=True)
    #     df['new_construction_x'].fillna(False, inplace=True)
    #     df['garage'].fillna(df['garage'].median(), inplace=True)
    #     df['stories'].fillna(df['stories'].median(), inplace=True)
    #     df['beds'].fillna(df['beds'].median(), inplace=True)
    #     df['baths'].fillna(df['baths'].median(), inplace=True)
    #     df['year_built'].fillna(df['year_built'].median(), inplace=True)
    #     df = df[df['lot_sqft'] < df['lot_sqft'].mean() + 3 * np.std(df['lot_sqft'])]
    #     df['lot_sqft'].fillna(df['lot_sqft'].median(), inplace=True)
    #     return df

    @log
    def analyze_df(self,df):
        nulls = cl(df.isnull().sum(), attrs=['bold'])
        description = df.describe()
        return nulls,description


