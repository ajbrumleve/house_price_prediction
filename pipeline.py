import pickle
import timeit
from zips import *
from logging_decorator import *
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import wx
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


@log
def get_realtor_object(state_abbr):
    r = RealtorScraper(page_numbers=206)
    try:
        df_obj = r.create_dataframe(state_abbr)
        r.df = df_obj[0]
        r.address_df = df_obj[1]
        r.zips_df = df_obj[2]
    except Exception as e:
        logging.error(e)
        return
    filename = 'RealtorObject.sav'
    pickle.dump(r, open(filename, 'wb'))
    return r

@log
def get_model(realtor_object):
    regr_model = Regression()
    regr_model.model = RandomForestRegressor()
    regr_model.train, regr_model.test = Regression.train_test_split(Regression, realtor_object.df)
    # Added code which cleaned the original data so this is redundant. It may be better to save cleaning until after
    # train-test split though to avoid data_leakage.
    # regr_model.train = Services.clean(Services, regr_model.train)
    # regr_model.test = Services.clean(Services, regr_model.test)
    train_nulls = Services.analyze_df(Services, regr_model.train)[0]
    train_description = Services.analyze_df(Services, regr_model.train)[1]
    test_nulls = Services.analyze_df(Services, regr_model.train)[0]
    test_description = Services.analyze_df(Services, regr_model.train)[1]
    logging.debug(f"training nulls - {train_nulls}\ntest nulls - {test_nulls}\ntrain description - "
                  f"{train_description}\ntest description - {test_description}")
    regr_model.X_train, regr_model.y_train = regr_model.x_and_y(regr_model.train)
    regr_model.X_test, regr_model.y_test = regr_model.x_and_y(regr_model.test)
    regr_model.model.fit(regr_model.X_train,regr_model.y_train)
    r.model = regr_model
    filename = f'{state_abbr}_realtor_model.sav'
    pickle.dump(r, open(filename, 'wb'))
    return regr_model

@log
def evaluate_model(model):
    print("The baseline is determined by using median values.")
    func_regr_model = model
    func_regr_model.predictions = func_regr_model.model.predict(func_regr_model.X_test)
    medians = np.full((len(func_regr_model.y_test),), np.median(func_regr_model.y_train))

    print("Baseline MSE - ",mean_squared_error(func_regr_model.y_test, np.log(medians), squared=False))
    print("MSE - ",mean_squared_error(func_regr_model.y_test, func_regr_model.predictions, squared=False))

    print("Baseline MAE - ",mean_absolute_error(func_regr_model.y_test, medians))
    print("MAE - ",mean_absolute_error(func_regr_model.y_test, func_regr_model.predictions))

    print("Baseline R2 - ",r2_score(func_regr_model.y_test, medians))
    print("R2 - ",r2_score(func_regr_model.y_test, func_regr_model.predictions))
    logging.info(f"Model R2 is {r2_score(func_regr_model.y_test, func_regr_model.predictions)}")

    # importance = func_regr_model.check_importance()

# @log
# def find_zip_encoding(realtor_object,zip):
#     encoded_zip = 0
#     for key, value in realtor_object.cat_maps["postal_code"].items():
#         if zip == value:
#             encoded_zip = key
#     return encoded_zip


@log
def predict_specific_address(realtor_object,model,zip,house_num):
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
                zip_scraper.df[col] = slice[col].copy().iloc[:,0]
        else:
            zip_scraper.df[col] = 0
    zip_scraper.df = zip_scraper.df[regr_model.train.columns]
    try:
        zip_scraper.real_price = zip_scraper.df['price'].values[0]
    except IndexError as e:
        print("The house number is not listed as for sale in the dataset")
        return
    zip_scraper.df = zip_scraper.df.drop(['price'], axis=1)
    for col in zip_scraper.df.columns.copy():
        zip_scraper.df[col].copy().fillna(r.df[col].median(),inplace = True)

    prediction = model.model.predict(zip_scraper.df)
    return zip_scraper.real_price, int(prediction[0])

# def filtered_list(realtor_object,model,zip_codes: list,min_bedrooms=3,min_sqft=1000,max_price=300000):
#         df = realtor_object.df
#         df = Services.clean(Services, df)
#
#         df = df[df["beds"] >= min_bedrooms]
#         df = df[df["sqft"] >= min_sqft]
#         df = df[df["price"] <= max_price]
#         encoded_zips = []
#         for zip in zip_codes:
#             encoded_zips.append(find_zip_encoding(realtor_object,str(zip)))
#         df = df[df["postal_code"].isin(encoded_zips)]
#         ids = df["id"]
#         df = df[model.train.columns]
#         predictions = model.model.predict(df.drop(["price"],axis=1))
#         df["prediction"] = predictions
#         df["actual-pred"] = df["price"] - df["prediction"]
#         df["id"] = ids
#         df = df.sort_values("actual-pred",ascending=True)
#         df = df.join(realtor_object.df[["address","postal_code","county"]])
#         df["postal_code"] = df.apply(lambda row : realtor_object.cat_maps["postal_code"][row["postal_code"]], axis = 1)
#         df["city"] = df.apply(lambda row : realtor_object.cat_maps["city"][row["city"]], axis = 1)
#         df = df[["price","address","city","postal_code","prediction","actual-pred"]]
#         df.to_csv("filtered_houses.csv")
#         return df

# def find_independent_features(df):
#     most_important_feats = []
#     cor = df.corr()
#     cor_target = abs(cor["price"])  # Selecting highly correlated features
#     relevant_features = cor_target[cor_target > 0.3]
#     relevant_labels = [relevant_features.index[i] for i in range(len(relevant_features))]
#     for i in range(len(relevant_features)):
#         feature = relevant_features.index[i]
#         if feature == "price":
#             continue
#         else:
#             new_cor_target = abs(cor[feature])#Selecting highly correlated features
#             new_relevant_features = new_cor_target[new_cor_target>0.5]
#             new_relevant_labels = [new_relevant_features.index[i] for i in range(len(new_relevant_features))]
#             correlated_feats = [x for x in new_relevant_labels if (x in relevant_labels) and (x != feature) and (x != "price")]
#             if len(correlated_feats) > 0:
#                 list_vals = []
#                 feature_value = cor['price'][feature]
#                 list_vals.append((feature,feature_value))
#                 for item in correlated_feats:
#                     value = cor['price'][item]
#                     list_vals.append((item,value))
#                     print(f"{feature} is correlated with {correlated_feats}")
#                 from operator import itemgetter
#                 max_feat = max(list_vals, key=itemgetter(1))[0]
#                 most_important_feats.append(max_feat)
#             else:
#                 most_important_feats.append(feature)
#     return set(most_important_feats)


def RFECVSelect(df,estimator=LinearRegression(), min_features_to_select=5, step=1, n_jobs=-1, scoring="neg_mean_absolute_error", cv=5):
    rfe_selector = RFECV(estimator=estimator, min_features_to_select=min_features_to_select, step=step,
                         n_jobs=n_jobs, scoring=scoring, cv=cv,verbose=2)
    ts = time.time()
    X = df.drop(['price'], axis=1)
    Y = df['price']

    # Split the training set into
    # training and validation set
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2)
    rfe_selector.fit(X_train, Y_train)
    X_train = rfe_selector.transform(X_train.copy())
    rfe_selector.get_support() #new_vector = [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id),
    #                                         avgAssists, avgTurnovers, tournamentSeed, getTourneyAppearances(team_id),
    #                                         totalPoss, totalfgmPerPoss, totalftmPerPoss, totaldrPerPoss, totalastPerPoss]

    logging.error(f"Finished RFECV in {time.time()-ts}")

    return rfe_selector
    print(rfe_selector.feature_names_in_[rfe_selector.support_ == False])

def find_deals(realtor_obj,model,min_beds,min_sqrt,max_price,counties,state_abbr):
    Zip_Obj = Zips(counties, state_abbr)
    Zip_Obj.get_zip_list()
    zip_codes = Zip_Obj.list_zips
    df = realtor_obj.df.copy()
    df_addresses = realtor_obj.address_df
    df_zips = realtor_obj.zips_df
    prices = df['price']
    X = df.drop(["price"],axis=1)

    predictions = model.model.predict(X)
    df['prediction'] = predictions
    df['price'] = prices
    df['price_diff'] = df['prediction'] - df['price']
    df['address'] = df_addresses.address
    df['city'] = df_addresses.city
    df['county'] = df_addresses.county
    df['postal_code'] = df_zips
    df = df.copy().sort_values("price_diff",ascending=False)
    df_filtered = df[(df["beds"] >= min_beds) &
                     (df["sqft"] >= min_sqrt) &
                     (df['postal_code'].isin(zip_codes)) &
                     (df['price'] <= max_price)]

    return df_filtered
def run():
    logger = logging.getLogger("test")
    logging.basicConfig(level=logging.INFO)
    log_info = logging.FileHandler('test-log.log')
    log_info.setLevel(logging.INFO)
    logging.getLogger('').addHandler(log_info)
    t_file = timeit.default_timer()
    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Starting program")
    # if scrape == True:
    #     t_section = timeit.default_timer()
    #     logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to scrape for dataset")
    #     r = get_realtor_object()
    #     logging.info(datetime.now().strftime(
    #         '%H:%M:%S.%f') + " - " + f"Dataset scraped in {timeit.default_timer() - t_section} seconds")
    # elif scrape == False:
    #     t_section = timeit.default_timer()
    #     logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to load dataset")
    #     try:
    #         filename = 'RealtorObject.sav'
    #         r = pickle.load(open(filename, 'rb'))
    #         logging.info(datetime.now().strftime(
    #             '%H:%M:%S.%f') + " - " + f"Dataset loaded")
    #     except FileNotFoundError:
    #         logging.error(datetime.now().strftime(
    #             '%H:%M:%S.%f') + " - " + "File not found, scraping for data")
    #         r = get_realtor_object()
    #         logging.info(datetime.now().strftime(
    #             '%H:%M:%S.%f') + " - " + f"Dataset scraped in {timeit.default_timer() - t_section} seconds")
    # else:
    #     logging.error(datetime.now().strftime(
    #         '%H:%M:%S.%f') + " - " + "Scrape parameter must be True or False.")
    #     return

    if prebuilt_model is not None:
        try:
            t_section = timeit.default_timer()
            logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to load model")
            filename = prebuilt_model
            regr_model = pickle.load(open(filename, 'rb'))
            logging.info(datetime.now().strftime(
                '%H:%M:%S.%f') + " - " + f"Model loaded")
        except FileNotFoundError:
            logging.error(datetime.now().strftime('%H:%M:%S.%f') + " - " + "File not found, building model")
            t_section = timeit.default_timer()
            logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to scrape for dataset")
            r = get_realtor_object(state_abbr)
            logging.info(datetime.now().strftime(
                '%H:%M:%S.%f') + " - " + f"Dataset scraped in {timeit.default_timer() - t_section} seconds")
            t_section = timeit.default_timer()
            logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to build model")
            regr_model = get_model(r)
            logging.info(datetime.now().strftime(
                '%H:%M:%S.%f') + " - " + f"Model built in {timeit.default_timer() - t_section} seconds")
    elif prebuilt_model == False:
        t_section = timeit.default_timer()
        logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to scrape for dataset")
        r = get_realtor_object()
        logging.info(datetime.now().strftime(
            '%H:%M:%S.%f') + " - " + f"Dataset scraped in {timeit.default_timer() - t_section} seconds")
        t_section = timeit.default_timer()
        logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to build model")
        regr_model = get_model(r)
        logging.info(datetime.now().strftime(
            '%H:%M:%S.%f') + " - " + f"Model built in {timeit.default_timer() - t_section} seconds")
    else:
        logging.error(datetime.now().strftime(
            '%H:%M:%S.%f') + " - " + "Prebuilt_model must be True or False.")
        return

    evaluate_model(regr_model)
    zip = input("what is the zip code?")
    house_num = input("What is the house number?")
    address_price = predict_specific_address(r,regr_model,zip,house_num)
    try:
        if address_price[0] > address_price[1]:
            print(f"The model predicts a price of {address_price[1]}. The actual price is {address_price[0]}. The house is {address_price[0]-address_price[1]} more expensive than the prediction.")
        elif address_price[0] < address_price[1]:
            print(f"The model predicts a price of {address_price[1]}. The actual price is {address_price[0]}. The house is {address_price[1] - address_price[0]} cheaper than the prediction.")
        else:
            print(f"The model predicts the exact price of {address_price[0]}")
    except TypeError as e:
        logging.error(e)


    # zip_scraper = RealtorZipScraper(page_numbers=10,columns=r.df.columns)
    # for key,value in r.cat_maps["postal_code"].items():
    #     if zip == value:
    #         encoded_zip = key
    # slice = zip_scraper.create_dataframe(zip)
    # slice = slice.rename(columns={"postal_code":encoded_zip})
    # slice[encoded_zip] = 1
    # slice = slice[slice["address"].str.contains(house_num)]
    # zip_scraper.df = pd.DataFrame()
    # for col in zip_scraper.df_columns:
    #     if col in slice.columns:
    #         zip_scraper.df[col] = slice[col]
    #     else:
    #         zip_scraper.df[col] = 0
    # zip_scraper.df = zip_scraper.df[regr_model.train.columns]
    # zip_scraper.real_price = zip_scraper.df['price'].values[0]
    # zip_scraper.df = zip_scraper.df.drop(['price'],axis=1)
    #

if __name__ == "__main__":
    logger = logging.getLogger("test")
    logging.basicConfig(level=logging.INFO)
    log_info = logging.FileHandler('test-log.log')
    log_info.setLevel(logging.INFO)
    logging.getLogger('').addHandler(log_info)
    t_file = timeit.default_timer()
    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + "Starting program")

    while True:
        prebuilt_model = input("Would you like to load a model? Y/N")
        if prebuilt_model == "Y":
            file_name = input("What is the file name of the model?")
            try:
                t_section = timeit.default_timer()
                logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to load model")
                r = pickle.load(open(file_name, 'rb'))
                regr_model = r.model
                logging.info(datetime.now().strftime(
                    '%H:%M:%S.%f') + " - " + f"Model loaded")
                break
            except:
                logging.info(datetime.now().strftime(
                    '%H:%M:%S.%f') + " - " + f"File not found")
            continue
        elif prebuilt_model == "N":
            while True:
                train_bool = input("Would you like to train a new model? Y/N")
                if train_bool == "Y":
                    state_abbr = input("What is the abbreviation of the state you want to scrape data from? e.g. MO")
                    t_section = timeit.default_timer()
                    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to scrape for dataset")
                    r = get_realtor_object(state_abbr)
                    logging.info(datetime.now().strftime(
                        '%H:%M:%S.%f') + " - " + f"Dataset scraped in {timeit.default_timer() - t_section} seconds")
                    t_section = timeit.default_timer()
                    logging.info(datetime.now().strftime('%H:%M:%S.%f') + " - " + f"Starting to build model")
                    regr_model = get_model(r)
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
        menu_1 = input("What would you like to do?\n1 - search address\n2 - create filtered table")
        if menu_1 == "1":
            zip_code = input("What is the zip code of the house? ")
            house_num = input("What is the house number of the house? ")
            address_price = predict_specific_address(r, regr_model, zip_code, house_num)
            try:
                if address_price[0] > address_price[1]:
                    print(
                        f"The model predicts a price of {address_price[1]}. The actual price is {address_price[0]}. The house is {address_price[0] - address_price[1]} more expensive than the prediction.")
                elif address_price[0] < address_price[1]:
                    print(
                        f"The model predicts a price of {address_price[1]}. The actual price is {address_price[0]}. The house is {address_price[1] - address_price[0]} cheaper than the prediction.")
                else:
                    print(f"The model predicts the exact price of {address_price[0]}")
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
                                f"The model predicts a price of {address_price[1]}. The actual price is {address_price[0]}. The house is {address_price[0] - address_price[1]} more expensive than the prediction.")
                        elif address_price[0] < address_price[1]:
                            print(
                                f"The model predicts a price of {address_price[1]}. The actual price is {address_price[0]}. The house is {address_price[1] - address_price[0]} cheaper than the prediction.")
                        else:
                            print(f"The model predicts the exact price of {address_price[0]}")
                    except TypeError as e:
                        logging.error(e)
                elif repeat =="N":
                    break
        elif menu_1 == "2":
            min_beds = float(input("What is the minimum number of bedrooms? "))
            min_sqft = float(input("What is the minimum number of square feet? "))
            max_price = float(input("What is the maximum asking price? "))

            counties = []
            while True:
                county = input('Add a county to search? If finished, type done.')
                if county == "done":
                    break
                else:
                    counties.append(county)
            state_abbr = input("What state are these counties in? eg MO ")

            filtered_df = find_deals(r,regr_model,min_beds,min_sqft,max_price,counties,state_abbr)
            out_file = input("Where do you want to save the csv?")
            filtered_df.to_csv(out_file)
        elif menu_1 == "break":
            break
        else:
            print("Choose 1 or 2.")

