import pickle

from pandas.errors import PerformanceWarning

from zips import *
from logging_decorator import *
from sklearn.feature_selection import RFECV
from realtor_com import RealtorScraper
from realtor_zip_search import RealtorZipScraper
from services import Services
from regression import Regression
import numpy as np
import logging
from datetime import datetime
import pandas as pd
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
@log
def get_realtor_object(state_abbr):
    """Retrieve a RealtorScraper object for a specific state.

        This function creates a RealtorScraper object and retrieves the necessary data
        by calling the `create_dataframe` method. It saves the resulting object to a file
        using pickle and returns the object.

        Args:
            state_abbr (str): The abbreviation of the state (e.g., 'CA', 'NY').

        Returns:
            RealtorScraper or None: The RealtorScraper object if successful, None otherwise.

        Example:
            r = get_realtor_object('CA')
    """
    r = RealtorScraper(page_numbers=206)
    try:
        r.df, r.address_df, r.zips_df = r.create_dataframe(state_abbr)

    except Exception as e:
        logging.error(e)
        return
    try:
        filename = f'models/{state_abbr}_RealtorObject.sav'
        pickle.dump(r, open(filename, 'wb'))
    except:
        print(Exception)

    return r


@log
def get_model(realtor_object, state_abbr, file_out=None, grid_search=False):
    """Retrieve a regression model using a RealtorScraper object.

        This function creates a regression model using the provided RealtorScraper object
        and saves the resulting model to a file using pickle. The `@log` decorator logs
        the function execution and timing.

        Args:
            realtor_object (RealtorScraper): The RealtorScraper object containing the data.
            state_abbr (str): The abbreviation of the state (e.g., 'CA', 'NY').
            file_out (str, optional): The output file name for the saved model. If not specified,
                it uses the default file name '{state_abbr}_realtor_model.sav'.

        Returns:
            Regression: The trained regression model.

        Example:
            regr_model = get_model(realtor_object, 'CA')
    """
    regr_model = Regression()

    regr_model.train, regr_model.test = Regression.train_test_split(regr_model, realtor_object.df)
    # Added code which cleaned the original data so this is redundant. It may be better to save cleaning until after
    # train-test split though to avoid data_leakage.
    # regr_model.train = Services.clean(Services, regr_model.train)
    # regr_model.test = Services.clean(Services, regr_model.test)

    # Check null values are handled
    train_nulls = Services.analyze_df(Services, regr_model.train)[0]
    train_description = Services.analyze_df(Services, regr_model.train)[1]
    test_nulls = Services.analyze_df(Services, regr_model.train)[0]
    test_description = Services.analyze_df(Services, regr_model.train)[1]
    logging.debug(f"training nulls - {train_nulls}\ntest nulls - {test_nulls}\ntrain description - "
                  f"{train_description}\ntest description - {test_description}")

    # Create X and Y vectors
    regr_model.X_train, regr_model.y_train = regr_model.x_and_y(regr_model.train)
    regr_model.X_test, regr_model.y_test = regr_model.x_and_y(regr_model.test)
    if grid_search:
        regr_model.model = regr_model.grid_search()
    else:
        regr_model.model = RandomForestRegressor(max_depth=None,n_estimators=200)
    regr_model.model.fit(regr_model.X_train, regr_model.y_train)
    realtor_object.model = regr_model
    if file_out is None:
        filename = f'models/{state_abbr}_realtor_model.sav'
    else:
        filename = file_out
    pickle.dump(regr_model, open(filename, 'wb'))
    return regr_model


@log
def evaluate_model(model):
    """Evaluate the performance of a regression model.

        This function evaluates the performance of the provided regression model
        by calculating various metrics such as Mean Squared Error (MSE), Mean Absolute
        Error (MAE), and R-squared (R2) scores. The `@log` decorator logs the function
        execution and timing.

        Args:
            model (Regression): The trained regression model.

        Returns:
            None

        Example:
            evaluate_model(regr_model)
    """
    print("The baseline is determined by using median values.")
    func_regr_model = model
    func_regr_model.predictions = func_regr_model.model.predict(func_regr_model.X_test)
    medians = np.full((len(func_regr_model.y_test),), np.median(func_regr_model.y_train))

    print("Baseline RMSE - ", mean_squared_error(func_regr_model.y_test, np.log(medians), squared=False))
    print("RMSE - ", mean_squared_error(func_regr_model.y_test, func_regr_model.predictions, squared=False))

    print("Baseline MAE - ", mean_absolute_error(func_regr_model.y_test, medians))
    print("MAE - ", mean_absolute_error(func_regr_model.y_test, func_regr_model.predictions))

    print("Baseline R2 - ", r2_score(func_regr_model.y_test, medians))
    print("R2 - ", r2_score(func_regr_model.y_test, func_regr_model.predictions))
    logging.info(f"Model R2 is {r2_score(func_regr_model.y_test, func_regr_model.predictions)}")

    # importance = func_regr_model.check_importance()


@log
def predict_specific_address(realtor_object, model, zip, house_num):
    """Predict the price for a specific address.

        This function predicts the price for a specific address based on the provided
        realtor object and trained regression model. The `@log` decorator logs the
        function execution and timing.

        Args:
            realtor_object (RealtorScraper): The realtor object containing the dataset.
            model (Regression): The trained regression model.
            zip (str): The ZIP code of the address.
            house_num (str): The house number of the address.

        Returns:
            tuple: A tuple containing the real price from the dataset and the predicted price.

        Example:
            predict_specific_address(realtor_object, regr_model, '12345', '123')
    """
    r = realtor_object

    zip_scraper = RealtorZipScraper(page_numbers=10, columns=r.df.columns.union(['address']))
    # encoded_zip = find_zip_encoding(r,zip)
    slice = zip_scraper.create_dataframe(zip)
    # slice = slice.rename(columns={"postal_code": encoded_zip})
    # slice[encoded_zip] = 1
    slice.dropna(subset=["address"], inplace=True)
    slice = slice[slice["address"].str.contains(str(house_num))]
    zip_scraper.df = pd.DataFrame()
    for col in zip_scraper.df_columns.copy():
        if col in slice.copy().columns:
            try:
                zip_scraper.df[col] = slice[col].copy()
            except:
                zip_scraper.df[col] = slice[col].copy().iloc[:, 0]
        else:
            zip_scraper.df[col] = 0
    zip_scraper.df = zip_scraper.df[model.train.columns]
    try:
        zip_scraper.real_price = zip_scraper.df['price'].values[0]
    except IndexError as e:
        print("The house number is not listed as for sale in the dataset")
        return "The house number is not listed as for sale in the dataset"
    zip_scraper.df = zip_scraper.df.drop(['price'], axis=1)
    for col in zip_scraper.df.columns.copy():
        zip_scraper.df[col].copy().fillna(r.df[col].median(), inplace=True)

    prediction = model.model.predict(zip_scraper.df)
    return zip_scraper.real_price, int(prediction[0])


def RFECVSelect(df, estimator=LinearRegression(), min_features_to_select=5, step=1, n_jobs=-1,
                scoring="neg_mean_absolute_error", cv=5):
    """Perform Recursive Feature Elimination with Cross-Validation (RFECV) on the given dataset.

        This function applies RFECV to select the optimal features for the given estimator.
        RFECV recursively eliminates features and selects the best subset of features based on
        the specified scoring metric and cross-validation.

        Args:
            df (DataFrame): The input dataset.
            estimator (estimator object, default=LinearRegression()): The estimator used in RFECV.
            min_features_to_select (int, default=5): The minimum number of features to select.
            step (int, default=1): The number of features to remove at each iteration.
            n_jobs (int, default=-1): The number of jobs to run in parallel (-1 uses all available processors).
            scoring (str, default="neg_mean_absolute_error"): The scoring metric used for feature evaluation.
            cv (int, default=5): The number of cross-validation folds.

        Returns:
            RFECV: The fitted RFECV object.

        Example:
            RFECVSelect(df, estimator=RandomForestRegressor(), min_features_to_select=10, step=2)
    """
    rfe_selector = RFECV(estimator=estimator, min_features_to_select=min_features_to_select, step=step,
                         n_jobs=n_jobs, scoring=scoring, cv=cv, verbose=2)
    ts = time.time()
    X = df.drop(['price'], axis=1)
    Y = df['price']

    # Split the training set into
    # training and validation set
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2)
    rfe_selector.fit(X_train, Y_train)
    X_train = rfe_selector.transform(X_train.copy())
    rfe_selector.get_support()

    logging.error(f"Finished RFECV in {time.time() - ts}")
    print(rfe_selector.feature_names_in_[rfe_selector.support_ == False])

    return rfe_selector


def find_deals(realtor_obj, model, min_beds, min_sqrt, max_price, counties, state_abbr):
    """Find real estate deals based on specified criteria.

        This function takes a Realtor object, a predictive model, and various criteria such as minimum
        number of bedrooms, minimum square footage, maximum price, counties, and state abbreviation.
        It filters the real estate data based on the criteria and returns a DataFrame of the filtered
        deals sorted by the price difference between the prediction and the actual price.

        Args:
            realtor_obj (RealtorScraper): The Realtor object containing real estate data.
            model (Regression): The predictive model used to make price predictions.
            min_beds (int): The minimum number of bedrooms required for a deal.
            min_sqrt (int): The minimum square footage required for a deal.
            max_price (int): The maximum price allowed for a deal.
            counties (list): A list of counties to include in the search.
            state_abbr (str): The abbreviation of the state to search in.

        Returns:
            DataFrame: A DataFrame of the filtered real estate deals sorted by price difference.

        Example:
            find_deals(realtor_obj, model, min_beds=2, min_sqrt=1000, max_price=200000, counties=['County1', 'County2'], state_abbr='CA')
    """
    Zip_Obj = Zips(counties, state_abbr)
    Zip_Obj.get_zip_list()
    zip_codes = Zip_Obj.list_zips
    df = realtor_obj.df.copy()
    df_addresses = realtor_obj.address_df
    df_zips = realtor_obj.zips_df
    prices = df['price']
    X = df.drop(["price"], axis=1)

    predictions = model.model.predict(X)
    df['prediction'] = predictions
    df['price'] = prices
    df['price_diff'] = df['prediction'] - df['price']
    df['address'] = df_addresses.address
    df['city'] = df_addresses.city
    df['county'] = df_addresses.county
    df['postal_code'] = df_zips
    df = df.copy().sort_values("price_diff", ascending=False)
    df_filtered = df[(df["beds"] >= min_beds) &
                     (df["sqft"] >= min_sqrt) &
                     (df['postal_code'].isin(zip_codes)) &
                     (df['price'] <= max_price)]

    return df_filtered


def run():
    def run():
        """Run the program to scrape, build, and evaluate the model.

        This function executes the main workflow of the program. It checks if a prebuilt model is available,
        and if not, it scrapes the dataset and builds a new model. Then, it evaluates the model's performance.
        After that, it prompts the user to enter a zip code and house number to predict the price of a specific address.
        Finally, it compares the predicted price with the actual price and prints the difference.

        Returns:
            None

        Example:
            run()
    """

    logger = logging.getLogger("test")
    logging.basicConfig(level=logging.INFO)
    log_info = logging.FileHandler('logs/test-log.log')
    log_info.setLevel(logging.INFO)
    logging.getLogger("pipeline")
    t_file = timeit.default_timer()
    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Starting program")

    while True:
        prebuilt_model = input("Would you like to load a model? Y/N\n")
        if prebuilt_model == "Y":
            state_abbr = input("What is the state abbreviation you want to load a model for? eg MO\n")
            try:
                t_section = timeit.default_timer()
                logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to load model")
                r = pickle.load(open(f"models/{state_abbr}_RealtorObject.sav", 'rb'))
                regr_model = pickle.load(open(f"models/{state_abbr}_realtor_model.sav", 'rb'))
                logging.info(datetime.now().strftime(
                    '%H:%M:%S.%f') + " - " + f"Model loaded")
                break
            except FileNotFoundError:
                logging.info(datetime.now().strftime(
                    '%H:%M:%S.%f') + " - " + f"File not found")
            continue
        elif prebuilt_model == "N":
            while True:
                train_bool = input("Would you like to train a new model? Y/N\n")
                if train_bool == "Y":
                    state_abbr = input("What is the abbreviation of the state you want to scrape data from? e.g. MO\n")
                    grid_search_prompt = input(
                        "Would you like to use grid search to tune your model? Processing time is significantly longer. Y/N\n")
                    if grid_search_prompt == "Y":
                        grid_search = True
                    else:
                        grid_search = False
                    t_section = timeit.default_timer()
                    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to scrape for dataset")
                    r = get_realtor_object(state_abbr)
                    logging.info(datetime.now().strftime(
                        '%H:%M:%S.%f') + " - " + f"Dataset scraped in {timeit.default_timer() - t_section} seconds")
                    t_section = timeit.default_timer()
                    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to build model")
                    regr_model = get_model(r, state_abbr, grid_search=grid_search)
                    evaluate_model(regr_model)
                    logging.info(datetime.now().strftime(
                        '%H:%M:%S.%f') + " - " + f"Model built in {timeit.default_timer() - t_section} seconds")
                    break
                elif train_bool == "N":
                    raise SystemExit(0)
                else:
                    print("Please choose Y or N.")
                continue
            break
        else:
            print("Enter Y or N.")

    while True:
        menu_1 = input("What would you like to do?\n1 - search address\n2 - create filtered table\n")
        if menu_1 == "1":
            zip_code = input("What is the zip code of the house? ")
            house_num = input("What is the house number of the house? ")
            address_price = predict_specific_address(r, regr_model, zip_code, house_num)
            try:
                if address_price[0] > address_price[1]:
                    print(
                        f"The model predicts a price of ${address_price[1]:,}. The actual price is ${address_price[0]:,}. The house is ${(address_price[0] - address_price[1]):,} more expensive than the prediction.")
                elif address_price[0] < address_price[1]:
                    print(
                        f"The model predicts a price of ${address_price[1]:,}. The actual price is ${address_price[0]:,}. The house is ${(address_price[1] - address_price[0]):,} cheaper than the prediction.")
                else:
                    print(f"The model predicts the exact price of ${address_price[0]:,}")
            except TypeError as e:
                logging.error(e)
            while True:
                repeat = input("Would you like to look up another house? Y/N ")
                if repeat == "Y":
                    zip_code = input("What is the zip code of the house? ")
                    house_num = input("What is the house number of the house? ")
                    address_price = predict_specific_address(r, regr_model, zip_code, house_num)
                    try:
                        if address_price[0] > address_price[1]:
                            print(
                                f"The model predicts a price of ${address_price[1]:,}. The actual price is ${address_price[0]:,}. The house is ${(address_price[0] - address_price[1]):,} more expensive than the prediction.")
                        elif address_price[0] < address_price[1]:
                            print(
                                f"The model predicts a price of ${address_price[1]:,}. The actual price is ${address_price[0]:,}. The house is ${(address_price[1] - address_price[0]):,} cheaper than the prediction.")
                        else:
                            print(f"The model predicts the exact price of ${address_price[0]:,}")
                    except TypeError as e:
                        logging.error(e)
                elif repeat == "N":
                    break
        elif menu_1 == "2":
            min_beds = float(input("What is the minimum number of bedrooms? "))
            min_sqft = float(input("What is the minimum number of square feet? "))
            max_price = float(input("What is the maximum asking price? "))

            counties = []
            while True:
                county = input('Add a county to search? If finished, type done. ')
                if county == "done":
                    break
                else:
                    counties.append(county)
            filtered_df = find_deals(r, regr_model, min_beds, min_sqft, max_price, counties, state_abbr)
            out_file = input("Where do you want to save the csv?")
            filtered_df.to_csv(out_file, index=False)
            finished = input("Are you finished? Y/N\n")
            if finished == "Y":
                raise SystemExit(0)
            else:
                continue
        elif menu_1 == "break":
            break
        else:
            print("Choose 1 or 2.")


if __name__ == "__main__":
    logger = logging.getLogger("test")
    logging.basicConfig(level=logging.INFO)
    log_info = logging.FileHandler('logs/test-log.log')
    log_info.setLevel(logging.INFO)
    logging.getLogger("pipeline")
    t_file = timeit.default_timer()
    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Starting program")

    while True:
        prebuilt_model = input("Would you like to load a model? Y/N\n")
        if prebuilt_model == "Y":
            state_abbr = input("What is the state abbreviation you want to load a model for? eg MO\n")
            try:
                t_section = timeit.default_timer()
                logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to load model")
                r = pickle.load(open(f"models/{state_abbr}_RealtorObject.sav", 'rb'))
                regr_model = pickle.load(open(f"models/{state_abbr}_realtor_model.sav", 'rb'))
                logging.info(datetime.now().strftime(
                    '%H:%M:%S.%f') + " - " + f"Model loaded")
                break
            except FileNotFoundError:
                logging.info(datetime.now().strftime(
                    '%H:%M:%S.%f') + " - " + f"File not found")
            continue
        elif prebuilt_model == "N":
            while True:
                train_bool = input("Would you like to train a new model? Y/N\n")
                if train_bool == "Y":
                    state_abbr = input("What is the abbreviation of the state you want to scrape data from? e.g. MO\n")
                    grid_search_prompt = input(
                        "Would you like to use grid search to tune your model? Processing time is significantly longer. Y/N\n")
                    if grid_search_prompt == "Y":
                        grid_search = True
                    else:
                        grid_search = False
                    t_section = timeit.default_timer()
                    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to scrape for dataset")
                    r = get_realtor_object(state_abbr)
                    print(r.df.head())
                    logging.info(datetime.now().strftime(
                        '%H:%M:%S.%f') + " - " + f"Dataset scraped in {timeit.default_timer() - t_section} seconds")
                    t_section = timeit.default_timer()
                    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to build model")
                    regr_model = get_model(r, state_abbr, grid_search=grid_search)
                    evaluate_model(regr_model)
                    logging.info(datetime.now().strftime(
                        '%H:%M:%S.%f') + " - " + f"Model built in {timeit.default_timer() - t_section} seconds")
                    break
                elif train_bool == "N":
                    raise SystemExit(0)
                else:
                    print("Please choose Y or N.")
                continue
            break
        else:
            print("Enter Y or N.")

    while True:
        menu_1 = input("What would you like to do?\n1 - search address\n2 - create filtered table\n")
        if menu_1 == "1":
            zip_code = input("What is the zip code of the house? ")
            house_num = input("What is the house number of the house? ")
            address_price = predict_specific_address(r, regr_model, zip_code, house_num)
            try:
                if address_price[0] > address_price[1]:
                    print(
                        f"The model predicts a price of ${address_price[1]:,}. The actual price is ${address_price[0]:,}. The house is ${(address_price[0] - address_price[1]):,} more expensive than the prediction.")
                elif address_price[0] < address_price[1]:
                    print(
                        f"The model predicts a price of ${address_price[1]:,}. The actual price is ${address_price[0]:,}. The house is ${(address_price[1] - address_price[0]):,} cheaper than the prediction.")
                else:
                    print(f"The model predicts the exact price of ${address_price[0]}")
            except TypeError as e:
                logging.error(e)
            while True:
                repeat = input("Would you like to look up another house? Y/N ")
                if repeat == "Y":
                    zip_code = input("What is the zip code of the house? ")
                    house_num = input("What is the house number of the house? ")
                    address_price = predict_specific_address(r, regr_model, zip_code, house_num)
                    try:
                        if address_price[0] > address_price[1]:
                            print(
                                f"The model predicts a price of ${address_price[1]:,}. The actual price is ${address_price[0]:,}. The house is {(address_price[0] - address_price[1]):,} more expensive than the prediction.")
                        elif address_price[0] < address_price[1]:
                            print(
                                f"The model predicts a price of ${address_price[1]:,}. The actual price is ${address_price[0]:,}. The house is {(address_price[1] - address_price[0]):,} cheaper than the prediction.")
                        else:
                            print(f"The model predicts the exact price of ${address_price[0]:,}")
                    except TypeError as e:
                        logging.error(e)
                elif repeat == "N":
                    break
        elif menu_1 == "2":
            min_beds = float(input("What is the minimum number of bedrooms? "))
            min_sqft = float(input("What is the minimum number of square feet? "))
            max_price = float(input("What is the maximum asking price? "))

            counties = []
            while True:
                county = input('Add a county to search? If finished, type done. ')
                if county == "done":
                    break
                else:
                    counties.append(county)
            filtered_df = find_deals(r, regr_model, min_beds, min_sqft, max_price, counties, state_abbr)
            out_file = input("Where do you want to save the csv?")
            filtered_df.to_csv(out_file, index=False)
            finished = input("Are you finished? Y/N\n")
            if finished == "Y":
                raise SystemExit(0)
            else:
                continue
        elif menu_1 == "break":
            break
        else:
            print("Choose 1 or 2.")
