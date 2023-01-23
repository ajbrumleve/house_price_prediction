from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from logging_decorator import *

class Regression:
    def __init__(self):
        self.model = LinearRegression()
        self.df = pd.DataFrame()

        pass

    @log
    def train_test_split(self,df):
        # X_var = df[
        #     ['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',
        #      'WoodDeckSF', 'OpenPorchSF']].values
        # y_var = df['SalePrice'].values

        train, test = train_test_split(df, test_size=0.2, random_state=0)
        return train, test

    @log
    def x_and_y(self,df):
        X_var = df.drop(["price"],axis=1)

        y_var = df['price'].values
        return X_var, y_var

    # Can potentially delete
    # @log
    # def check_importance(self):
    #     feature_importance = {}
    #     importance_list = self.model.feature_importances_
    #     for i in range(865):
    #         feature_importance[self.X_train.columns[i]] = importance_list[i]
    #     sorted_dict = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    #     return sorted_dict
