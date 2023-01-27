import requests
import json
import pandas as pd
from scipy import stats
import numpy as np
from bs4 import BeautifulSoup
from termcolor import colored as cl # text customization
import matplotlib.pyplot as plt
import seaborn as sns
from logging_decorator import *

class RealtorScraper:
    def __init__(self, page_numbers: int) -> None:
        self.page_numbers = page_numbers
        self.cat_maps = {}

    def send_request(self, page_number: int, offset_parameter: int, state_abbr: str) -> dict:

        url = "https://www.realtor.com/api/v1/hulk?client_id=rdc-x&schema=vesta"
        headers = {"content-type": "application/json"}

        body = r'{"query":"\n\nquery ConsumerSearchMainQuery($query: HomeSearchCriteria!, $limit: Int, $offset: Int, $sort: [SearchAPISort], $sort_type: SearchSortType, $client_data: JSON, $bucket: SearchAPIBucket)\n{\n  home_search: home_search(query: $query,\n    sort: $sort,\n    limit: $limit,\n    offset: $offset,\n    sort_type: $sort_type,\n    client_data: $client_data,\n    bucket: $bucket,\n  ){\n    count\n    total\n    results {\n      property_id\n      list_price\n      primary\n      primary_photo (https: true){\n        href\n      }\n      source {\n        id\n        agents{\n          office_name\n        }\n        type\n        spec_id\n        plan_id\n      }\n      community {\n        property_id\n        description {\n          name\n        }\n        advertisers{\n          office{\n            hours\n            phones {\n              type\n              number\n            }\n          }\n          builder {\n            fulfillment_id\n          }\n        }\n      }\n      products {\n        brand_name\n        products\n      }\n      listing_id\n      matterport\n      virtual_tours{\n        href\n        type\n      }\n      status\n      permalink\n      price_reduced_amount\n      other_listings{rdc {\n      listing_id\n      status\n      listing_key\n      primary\n    }}\n      description{\n        beds\n        baths\n        baths_full\n        baths_half\n        baths_1qtr\n        baths_3qtr\n        garage\n        stories\n        type\n        sub_type\n        lot_sqft\n        sqft\n        year_built\n        sold_price\n        sold_date\n        name\n      }\n      location{\n        street_view_url\n        address{\n          line\n          postal_code\n          state\n          state_code\n          city\n          coordinate {\n            lat\n            lon\n          }\n        }\n        county {\n          name\n          fips_code\n        }\n      }\n      tax_record {\n        public_record_id\n      }\n      lead_attributes {\n        show_contact_an_agent\n        opcity_lead_attributes {\n          cashback_enabled\n          flip_the_market_enabled\n        }\n        lead_type\n        ready_connect_mortgage {\n          show_contact_a_lender\n          show_veterans_united\n        }\n      }\n      open_houses {\n        start_date\n        end_date\n        description\n        methods\n        time_zone\n        dst\n      }\n      flags{\n        is_coming_soon\n        is_pending\n        is_foreclosure\n        is_contingent\n        is_new_construction\n        is_new_listing (days: 14)\n        is_price_reduced (days: 30)\n        is_plan\n        is_subdivision\n      }\n      list_date\n      last_update_date\n      coming_soon_date\n      photos(limit: 2, https: true){\n        href\n      }\n      tags\n      branding {\n        type\n        photo\n        name\n      }\n    }\n  }\n}","variables":{"query":{"status":["for_sale","ready_to_build"],"primary":true,"state_code":"MO"},"client_data":{"device_data":{"device_type":"web"},"user_data":{"last_view_timestamp":-1}},"limit":42,"offset":42,"zohoQuery":{"silo":"search_result_page","location":"Missouri","property_status":"for_sale","filters":{"radius":null},"page_index":"2"},"sort_type":"relevant","geoSupportedSlug":"","resetMap":"2022-12-15T20:04:44.616Z0.226202430038297","by_prop_type":["home"]},"operationName":"ConsumerSearchMainQuery","callfrom":"SRP","nrQueryType":"MAIN_SRP","visitor_id":"83307539-de0a-4311-8ea6-05c47e404dc0","isClient":true,"seoPayload":{"asPath":"/realestateandhomes-search/Missouri/pg-2","pageType":{"silo":"search_result_page","status":"for_sale"},"county_needed_for_uniq":false}}'

        json_body = json.loads(body)

        json_body["variables"]["page_index"] = page_number
        json_body["seoPayload"] = page_number
        json_body["variables"]["offset"] = offset_parameter
        json_body["variables"]["query"]['state_code'] = state_abbr


        r = requests.post(url=url, json=json_body, headers=headers)
        json_data = r.json()
        return json_data

    def extract_features(self, entry: dict) -> dict:
        feature_dict = {
            "id": entry["property_id"],
            "price": entry["list_price"],
            "beds": entry["description"]["beds"],
            "baths": entry["description"]["baths"],
            "garage": entry["description"]["garage"],
            "stories": entry["description"]["stories"],
            "house_type": entry["description"]["type"],
            "lot_sqft": entry["description"]["lot_sqft"],
            "sqft": entry["description"]["sqft"],
            "price_reduced": entry["flags"]["is_price_reduced"],
            "foreclosure": entry["flags"]["is_foreclosure"],
            "new_construction": entry["flags"]["is_new_construction"],
            "new_listing": entry["flags"]["is_new_listing"],
            "subdivision": entry["flags"]["is_subdivision"],
            "year_built": entry["description"]["year_built"],
            "address": entry["location"]["address"]["line"],
            "postal_code": entry["location"]["address"]["postal_code"],
            "state": entry["location"]["address"]["state_code"],
            "city": entry["location"]["address"]["city"],
            "tags": entry["tags"]
        }

        if entry["location"]["address"]["coordinate"]:
            feature_dict.update({"lat": entry["location"]["address"]["coordinate"]["lat"]})
            feature_dict.update({"lon": entry["location"]["address"]["coordinate"]["lon"]})

        if entry["location"]["county"]:
            feature_dict.update({"county": entry["location"]["county"]["name"]})

        return feature_dict

    @log
    def parse_json_data(self, state_abbr: str = "MO") -> list:
        offset_parameter = 0

        feature_dict_list = []

        for i in range(1, self.page_numbers):
            json_data = self.send_request(page_number=i, offset_parameter=offset_parameter,state_abbr=state_abbr)
            offset_parameter += 42

            for entry in json_data["data"]["home_search"]["results"]:
                feature_dict = self.extract_features(entry)
                feature_dict_list.append(feature_dict)

        return feature_dict_list

    @log
    def create_dataframe(self, state_abbr: str) -> pd.DataFrame:
        feature_dict_list = self.parse_json_data(state_abbr)
        df = pd.DataFrame(feature_dict_list)
        # select house_type = single_family, price<1000000, beds<7, baths<10, garage < 5
        # drop state, lat, lon
        df_new = df.copy()
        df_new['price'] = df_new['price'].fillna(df_new['price'].median())
        df_new['beds'] = df_new['beds'].fillna(df_new['beds'].median())
        df_new['baths'] = df_new['baths'].fillna(df_new['baths'].median())
        df_new['garage'] = df_new['garage'].fillna(df_new['garage'].median())
        df_new['stories'] = df_new['stories'].fillna(df_new['stories'].median())
        df_new = df_new[df_new['price'] < 1000000]
        df_new = df_new[df_new['beds'] < 7]
        df_new = df_new[df_new['baths'] < 10]
        df_new = df_new[df_new['garage'] < 5]
        df_new = df_new[df_new['house_type'] == 'single_family']

        df_new = df_new[df_new['lot_sqft'] < 50000000]

        df_new['year_built'] = df_new['year_built'].fillna(df_new['year_built'].median())
        df_new['sqft'] = df_new['sqft'].fillna(df_new['sqft'].median())
        df_new['foreclosure'] = df_new['foreclosure'].fillna(False)
        df_new['new_construction'] = df_new['new_construction'].fillna(False)
        df_new['price_reduced'] = df_new['price_reduced'].fillna(False)
        df_new.drop(['id', 'house_type', 'subdivision', 'state', 'lat', 'lon'], axis=1,
                    inplace=True)
        df_new = df_new.dropna()
        address_df = df_new.copy()[['address',"city","county"]]
        zip_df = df_new.copy()['postal_code']
        df_new.drop(['address','city','county'], axis=1, inplace=True)
        categorical_cols = ["postal_code"]
        df_new[categorical_cols] = df_new[categorical_cols].astype('category')
        from sklearn.preprocessing import OneHotEncoder

        s = (df_new.dtypes == 'category')
        object_cols = list(s[s].index)
        print("Categorical variables:")
        print(object_cols)
        print('No. of. categorical features: ', len(object_cols))

        OH_encoder = OneHotEncoder(sparse=False)
        OH_cols = pd.DataFrame(OH_encoder.fit_transform(df_new[object_cols]))
        OH_cols.index = df_new.index
        OH_cols.columns = OH_encoder.get_feature_names()
        dummy_df = pd.get_dummies(df_new['tags'].explode()).groupby(level=0).sum()
        df_final = df_new.drop(object_cols, axis=1)
        df_final = df_final.drop('tags', axis=1)
        df_final = pd.concat([df_final, dummy_df], axis=1)
        return df_final, address_df, zip_df


if __name__ == "__main__":
    r = RealtorScraper(page_numbers=300)
    df = r.create_dataframe()[0]
    address_df = r.create_dataframe()[1]
    df.dropna(subset = ['beds', 'baths', 'stories','sqft','address'],inplace = True)
    df['subdivision'].fillna(False,inplace = True)
    df['foreclosure'].fillna(False,inplace = True)
    df['price_reduced'].fillna(False,inplace = True)
    df['new_construction'].fillna(False,inplace = True)
    df['garage'].fillna(df['garage'].mean(),inplace = True)
    df['year_built'].fillna(df['year_built'].mean(),inplace = True)
    df = df[df['lot_sqft'] < df['lot_sqft'].mean()+3*np.std(df['lot_sqft'])]
    df['lot_sqft'].fillna(df['lot_sqft'].median(),inplace = True)
    df = df[df['price'] < 500000]
    nulls = cl(df.isnull().sum(), attrs=['bold'])
    description = df.describe()
    print(cl(df.dtypes, attrs=['bold']))
    cols = r.df.columns[r.df.dtypes.eq('float64')]
    r.df[cols] = r.df[cols].apply(pd.to_numeric, errors='coerce')
    for col in cols:
        df[col] = df[col].astype('int64')
        print(col)

def explore_data():
    obj = (df.dtypes == 'object')
    object_cols = list(obj[obj].index)
    print("Categorical variables:", len(object_cols))
    int_ = (df.dtypes == 'int')
    num_cols = list(int_[int_].index)
    print("Integer variables:", len(num_cols))
    fl = (df.dtypes == 'float')
    fl_cols = list(fl[fl].index)
    print("Float variables:", len(fl_cols))
    ct = (df.dtypes == 'category')
    ct_cols = list(ct[ct].index)
    print("Float variables:", len(ct_cols))

    unique_values = []
    for col in ct_cols:
        unique_values.append(df[col].unique().size)
    plt.figure(figsize=(10, 6))
    plt.title('No. Unique values of Categorical Features')
    plt.xticks(rotation=90)
    sns.barplot(x=ct_cols, y=unique_values)

    plt.figure(figsize=(18, 36))
    plt.title('Categorical Features: Distribution')
    plt.xticks(rotation=90)
    index = 1

    for col in ct_cols[:4]:
        y = df[col].value_counts()
        plt.subplot(11, 4, index)
        plt.xticks(rotation=90)
        sns.barplot(x=list(y.index), y=y)
        index += 1

    sns.distplot(df['price'])

    var = 'year_built'
    data = pd.concat([df_new['price'], df_new[var]], axis=1)
    data.plot.scatter(x=var, y='price', ylim=(0, 800000), s=32)

def clean_data(df):
    df_new = df.copy()
    df_new['price'] = df_new['price'].fillna(df_new['price'].median())
    df_new['beds'] = df_new['beds'].fillna(df_new['beds'].median())
    df_new['baths'] = df_new['baths'].fillna(df_new['baths'].median())
    df_new['garage'] = df_new['garage'].fillna(df_new['garage'].median())
    df_new['stories'] = df_new['stories'].fillna(df_new['stories'].median())
    df_new = df_new[df_new['price'] < 1000000]
    df_new = df_new[df_new['beds'] < 7]
    df_new = df_new[df_new['baths'] < 10]
    df_new = df_new[df_new['garage'] < 5]
    df_new = df_new[df_new['house_type'] == 'single_family']

    df_new = df_new[df_new['lot_sqft'] < 50000000]

    df_new['year_built'] = df_new['year_built'].fillna(df_new['year_built'].median())
    df_new['sqft'] = df_new['sqft'].fillna(df_new['sqft'].median())
    df_new['foreclosure'] = df_new['foreclosure'].fillna(False)
    df_new['new_construction'] = df_new['new_construction'].fillna(False)
    df_new['price_reduced'] = df_new['price_reduced'].fillna(False)
    df_new.drop(['id','house_type','subdivision','state','lat','lon','address', "city", "county"],axis = 1,inplace = True)
    df_new = df_new.dropna()
    categorical_cols = [ "postal_code"]
    df_new[categorical_cols] = df_new[categorical_cols].astype('category')
    from sklearn.preprocessing import OneHotEncoder

    s = (df_new.dtypes == 'category')
    object_cols = list(s[s].index)
    print("Categorical variables:")
    print(object_cols)
    print('No. of. categorical features: ', len(object_cols))

    OH_encoder = OneHotEncoder(sparse=False)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(df_new[object_cols]))
    OH_cols.index = df_new.index
    OH_cols.columns = OH_encoder.get_feature_names()
    dummy_df = pd.get_dummies(df_new['tags'].explode()).groupby(level=0).sum()
    df_final = df_new.drop(object_cols, axis=1)
    df_final = df_final.drop('tags', axis=1)
    df_final = pd.concat([df_final, dummy_df], axis=1)
    return df_new

def model(df):
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import svm
    from sklearn.svm import SVC

    X = df.drop(['price'], axis=1)
    Y = df['price']

    # Split the training set into
    # training and validation set
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size = 0.8, test_size = 0.2)

    from sklearn.linear_model import LinearRegression

    model_SVR = svm.SVR()
    model_SVR.fit(X_train, Y_train)
    Y_pred = model_SVR.predict(X_valid)
    print("SVR - ",mean_absolute_percentage_error(Y_valid, Y_pred))

    model_RFR = RandomForestRegressor(n_estimators=10)
    model_RFR.fit(X_train, Y_train)
    Y_pred = model_RFR.predict(X_valid)
    print("RF - ",mean_absolute_percentage_error(Y_valid, Y_pred))

    model_LR = LinearRegression()
    model_LR.fit(X_train, Y_train)
    Y_pred = model_LR.predict(X_valid)
    print("LR - ",mean_absolute_percentage_error(Y_valid, Y_pred))