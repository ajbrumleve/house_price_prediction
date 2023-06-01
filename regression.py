from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from logging_decorator import *

class Regression:
    def __init__(self):
        """
        Initialize a Regression object.

        This constructor initializes the model as LinearRegression and the DataFrame as an empty DataFrame.
        """
        self.model = LinearRegression()
        self.df = pd.DataFrame()

        pass

    @log
    def train_test_split(self,df):
        """
        Splits the provided DataFrame into train and test sets.

        Parameters:
            df (pd.DataFrame): The DataFrame to be split into train and test sets.

        Returns:
            tuple: A tuple containing the train and test DataFrames.

        Note:
            This function assumes that the DataFrame has already been preprocessed and contains the necessary columns
            for model training and testing.

        The @log decorator is used to log information about the function call and its results.
        """
        # X_var = df[
        #     ['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',
        #      'WoodDeckSF', 'OpenPorchSF']].values
        # y_var = df['SalePrice'].values

        train, test = train_test_split(df, test_size=0.2, random_state=0)
        return train, test

    @log
    def x_and_y(self,df):
        """
        Extracts the features (X) and target variable (y) from the provided DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame from which to extract features and target variable.

        Returns:
            tuple: A tuple containing the features (X) and target variable (y).

        Note:
            This function assumes that the DataFrame has already been preprocessed and contains the necessary columns
            for feature extraction and target variable.

        The @log decorator is used to log information about the function call and its results.
        """
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
