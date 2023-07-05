from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

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
    def train_test_split(self, df):
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
    def x_and_y(self, df):
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
        X_var = df.drop(["price"], axis=1)

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

    @log
    def grid_search(self):
        """
        Perform grid search to find the best model with optimized hyperparameters.

        This function performs grid search on multiple regression algorithms to find the best model
        with optimized hyperparameters based on the provided training data. The algorithms and their
        corresponding parameter grids are defined within the function.

        Returns:
            best_model: The best model found during the grid search process.

        Example:
            regr_model = grid_search()
        """
        algorithms = [
            ('Random Forest', RandomForestRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}),
            ('XGBoost', XGBRegressor(),
             {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]})
        ]
        best_models = []
        for name, model, param_grid in algorithms:
            grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=5, verbose=3)
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            best_models.append((name, best_model, grid_search.best_params_, r2))
            print(f"{name}: Best parameters - {grid_search.best_params_}, MSE: {mse}, R2: {r2}")

        # Find the algorithm with the highest R2 score
        best_model_info = max(best_models, key=lambda x: x[3])
        best_model_name, best_model, best_params, best_r2 = best_model_info
        print(f"The selected model is {best_model_name} with params: {best_params} with an R2 of {best_r2}")
        return best_model
