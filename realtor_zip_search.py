import requests
import json
import pandas as pd
from scipy import stats
import numpy as np
from bs4 import BeautifulSoup
from termcolor import colored as cl # text customization

class RealtorZipScraper:
    def __init__(self, page_numbers: int, columns: list) -> None:
        self.page_numbers = page_numbers
        self.df_columns = columns

    #Can potentially delete
    # def send_request(self, page_number: int, offset_parameter: int) -> dict:
    #
    #     url = "https://www.realtor.com/api/v1/hulk?client_id=rdc-x&schema=vesta"
    #     headers = {"content-type": "application/json"}
    #
    #     body = r'{"query":"\n\nquery ConsumerSearchMainQuery($query: HomeSearchCriteria!, $limit: Int, $offset: Int, $sort: [SearchAPISort], $sort_type: SearchSortType, $client_data: JSON, $bucket: SearchAPIBucket)\n{\n  home_search: home_search(query: $query,\n    sort: $sort,\n    limit: $limit,\n    offset: $offset,\n    sort_type: $sort_type,\n    client_data: $client_data,\n    bucket: $bucket,\n  ){\n    count\n    total\n    results {\n      property_id\n      list_price\n      primary\n      primary_photo (https: true){\n        href\n      }\n      source {\n        id\n        agents{\n          office_name\n        }\n        type\n        spec_id\n        plan_id\n      }\n      community {\n        property_id\n        description {\n          name\n        }\n        advertisers{\n          office{\n            hours\n            phones {\n              type\n              number\n            }\n          }\n          builder {\n            fulfillment_id\n          }\n        }\n      }\n      products {\n        brand_name\n        products\n      }\n      listing_id\n      matterport\n      virtual_tours{\n        href\n        type\n      }\n      status\n      permalink\n      price_reduced_amount\n      other_listings{rdc {\n      listing_id\n      status\n      listing_key\n      primary\n    }}\n      description{\n        beds\n        baths\n        baths_full\n        baths_half\n        baths_1qtr\n        baths_3qtr\n        garage\n        stories\n        type\n        sub_type\n        lot_sqft\n        sqft\n        year_built\n        sold_price\n        sold_date\n        name\n      }\n      location{\n        street_view_url\n        address{\n          line\n          postal_code\n          state\n          state_code\n          city\n          coordinate {\n            lat\n            lon\n          }\n        }\n        county {\n          name\n          fips_code\n        }\n      }\n      tax_record {\n        public_record_id\n      }\n      lead_attributes {\n        show_contact_an_agent\n        opcity_lead_attributes {\n          cashback_enabled\n          flip_the_market_enabled\n        }\n        lead_type\n        ready_connect_mortgage {\n          show_contact_a_lender\n          show_veterans_united\n        }\n      }\n      open_houses {\n        start_date\n        end_date\n        description\n        methods\n        time_zone\n        dst\n      }\n      flags{\n        is_coming_soon\n        is_pending\n        is_foreclosure\n        is_contingent\n        is_new_construction\n        is_new_listing (days: 14)\n        is_price_reduced (days: 30)\n        is_plan\n        is_subdivision\n      }\n      list_date\n      last_update_date\n      coming_soon_date\n      photos(limit: 2, https: true){\n        href\n      }\n      tags\n      branding {\n        type\n        photo\n        name\n      }\n    }\n  }\n}","variables":{"query":{"status":["for_sale","ready_to_build"],"primary":true,"state_code":"MO"},"client_data":{"device_data":{"device_type":"web"},"user_data":{"last_view_timestamp":-1}},"limit":42,"offset":42,"zohoQuery":{"silo":"search_result_page","location":"Missouri","property_status":"for_sale","filters":{"radius":null},"page_index":"2"},"sort_type":"relevant","geoSupportedSlug":"","resetMap":"2022-12-15T20:04:44.616Z0.226202430038297","by_prop_type":["home"]},"operationName":"ConsumerSearchMainQuery","callfrom":"SRP","nrQueryType":"MAIN_SRP","visitor_id":"83307539-de0a-4311-8ea6-05c47e404dc0","isClient":true,"seoPayload":{"asPath":"/realestateandhomes-search/Missouri/pg-2","pageType":{"silo":"search_result_page","status":"for_sale"},"county_needed_for_uniq":false}}'
    #
    #     json_body = json.loads(body)
    #
    #     json_body["variables"]["page_index"] = page_number
    #     json_body["seoPayload"] = page_number
    #     json_body["variables"]["offset"] = offset_parameter
    #
    #
    #     r = requests.post(url=url, json=json_body, headers=headers)
    #     json_data = r.json()
    #     return json_data

    def search_addess(self, page_number: int, offset_parameter: int, zip) -> dict:
        # page_number= 1
        # offset_parameter = 42
        url = "https://www.realtor.com/api/v1/hulk?client_id=rdc-x&schema=vesta"
        headers = {"content-type": "application/json"}
        body = r'{"query":"\n\nquery ConsumerSearchMainQuery($query: HomeSearchCriteria!, $limit: Int, $offset: Int, $sort: [SearchAPISort], $sort_type: SearchSortType, $client_data: JSON, $bucket: SearchAPIBucket)\n{\n  home_search: home_search(query: $query,\n    sort: $sort,\n    limit: $limit,\n    offset: $offset,\n    sort_type: $sort_type,\n    client_data: $client_data,\n    bucket: $bucket,\n  ){\n    count\n    total\n    results {\n      property_id\n      list_price\n      primary\n      primary_photo (https: true){\n        href\n      }\n      source {\n        id\n        agents{\n          office_name\n        }\n        type\n        spec_id\n        plan_id\n      }\n      community {\n        property_id\n        description {\n          name\n        }\n        advertisers{\n          office{\n            hours\n            phones {\n              type\n              number\n            }\n          }\n          builder {\n            fulfillment_id\n          }\n        }\n      }\n      products {\n        brand_name\n        products\n      }\n      listing_id\n      matterport\n      virtual_tours{\n        href\n        type\n      }\n      status\n      permalink\n      price_reduced_amount\n      other_listings{rdc {\n      listing_id\n      status\n      listing_key\n      primary\n    }}\n      description{\n        beds\n        baths\n        baths_full\n        baths_half\n        baths_1qtr\n        baths_3qtr\n        garage\n        stories\n        type\n        sub_type\n        lot_sqft\n        sqft\n        year_built\n        sold_price\n        sold_date\n        name\n      }\n      location{\n        street_view_url\n        address{\n          line\n          postal_code\n          state\n          state_code\n          city\n          coordinate {\n            lat\n            lon\n          }\n        }\n        county {\n          name\n          fips_code\n        }\n      }\n      tax_record {\n        public_record_id\n      }\n      lead_attributes {\n        show_contact_an_agent\n        opcity_lead_attributes {\n          cashback_enabled\n          flip_the_market_enabled\n        }\n        lead_type\n        ready_connect_mortgage {\n          show_contact_a_lender\n          show_veterans_united\n        }\n      }\n      open_houses {\n        start_date\n        end_date\n        description\n        methods\n        time_zone\n        dst\n      }\n      flags{\n        is_coming_soon\n        is_pending\n        is_foreclosure\n        is_contingent\n        is_new_construction\n        is_new_listing (days: 14)\n        is_price_reduced (days: 30)\n        is_plan\n        is_subdivision\n      }\n      list_date\n      last_update_date\n      coming_soon_date\n      photos(limit: 2, https: true){\n        href\n      }\n      tags\n      branding {\n        type\n        photo\n        name\n      }\n    }\n  }\n}","variables":{"query":{"status":["for_sale","ready_to_build"],"primary":true,"search_location":{"location":"63017, Chesterfield, MO"}},"client_data":{"device_data":{"device_type":"web"},"user_data":{"last_view_timestamp":-1}},"limit":42,"offset":42,"zohoQuery":{"silo":"search_result_page","location":"63017, Chesterfield, MO","property_status":"for_sale","filters":{},"page_index":"2"},"sort_type":"relevant","geoSupportedSlug":"63017","by_prop_type":["home"]},"operationName":"ConsumerSearchMainQuery","callfrom":"SRP","nrQueryType":"MAIN_SRP","visitor_id":"83307539-de0a-4311-8ea6-05c47e404dc0","isClient":true,"seoPayload":{"asPath":"/realestateandhomes-search/63017/pg-2","pageType":{"silo":"search_result_page","status":"for_sale"},"county_needed_for_uniq":false}}'
        # body = r'{"query":"\n\nquery ConsumerSearchMainQuery($query: HomeSearchCriteria!, $limit: Int, $offset: Int, $sort: [SearchAPISort], $sort_type: SearchSortType, $client_data: JSON, $bucket: SearchAPIBucket)\n{\n  home_search: home_search(query: $query,\n    sort: $sort,\n    limit: $limit,\n    offset: $offset,\n    sort_type: $sort_type,\n    client_data: $client_data,\n    bucket: $bucket,\n  ){\n    count\n    total\n    results {\n      property_id\n      list_price\n      primary\n      primary_photo (https: true){\n        href\n      }\n      source {\n        id\n        agents{\n          office_name\n        }\n        type\n        spec_id\n        plan_id\n      }\n      community {\n        property_id\n        description {\n          name\n        }\n        advertisers{\n          office{\n            hours\n            phones {\n              type\n              number\n            }\n          }\n          builder {\n            fulfillment_id\n          }\n        }\n      }\n      products {\n        brand_name\n        products\n      }\n      listing_id\n      matterport\n      virtual_tours{\n        href\n        type\n      }\n      status\n      permalink\n      price_reduced_amount\n      other_listings{rdc {\n      listing_id\n      status\n      listing_key\n      primary\n    }}\n      description{\n        beds\n        baths\n        baths_full\n        baths_half\n        baths_1qtr\n        baths_3qtr\n        garage\n        stories\n        type\n        sub_type\n        lot_sqft\n        sqft\n        year_built\n        sold_price\n        sold_date\n        name\n      }\n      location{\n        street_view_url\n        address{\n          line\n          postal_code\n          state\n          state_code\n          city\n          coordinate {\n            lat\n            lon\n          }\n        }\n        county {\n          name\n          fips_code\n        }\n      }\n      tax_record {\n        public_record_id\n      }\n      lead_attributes {\n        show_contact_an_agent\n        opcity_lead_attributes {\n          cashback_enabled\n          flip_the_market_enabled\n        }\n        lead_type\n        ready_connect_mortgage {\n          show_contact_a_lender\n          show_veterans_united\n        }\n      }\n      open_houses {\n        start_date\n        end_date\n        description\n        methods\n        time_zone\n        dst\n      }\n      flags{\n        is_coming_soon\n        is_pending\n        is_foreclosure\n        is_contingent\n        is_new_construction\n        is_new_listing (days: 14)\n        is_price_reduced (days: 30)\n        is_plan\n        is_subdivision\n      }\n      list_date\n      last_update_date\n      coming_soon_date\n      photos(limit: 2, https: true){\n        href\n      }\n      tags\n      branding {\n        type\n        photo\n        name\n      }\n    }\n  }\n}","variables":{"query":{"status":["for_sale","ready_to_build"],"primary":true,"search_location":{"location":"63043, Chesterfield, MO"}},"client_data":{"device_data":{"device_type":"web"},"user_data":{"last_view_timestamp":-1}},"limit":42,"offset":0,"zohoQuery":{"silo":"search_result_page","location":"63043, Maryland Heights, MO","property_status":"for_sale","filters":{}},"sort_type":"relevant","geoSupportedSlug":"63043","by_prop_type":["home"]},"operationName":"ConsumerSearchMainQuery","callfrom":"SRP","nrQueryType":"MAIN_SRP","visitor_id":"83307539-de0a-4311-8ea6-05c47e404dc0","isClient":true,"seoPayload":{"asPath":"/realestateandhomes-search/63043","pageType":{"silo":"search_result_page","status":"for_sale"},"county_needed_for_uniq":false}}'
        body = body.replace("63017",zip)
        json_body = json.loads(body)

        json_body["variables"]["page_index"] = page_number
        json_body['seoPayload']['asPath'] = json_body['seoPayload']['asPath'].replace("pg-2",f"pg-{str(page_number)}")
        json_body["variables"]["offset"] = offset_parameter

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

    def parse_json_data(self,zip) -> list:
        offset_parameter = 0

        feature_dict_list = []
        search_complete = False
        for i in range(1, self.page_numbers):
            if search_complete == False:
                json_data = self.search_addess(page_number=i, offset_parameter=offset_parameter,zip=zip)
                if json_data['data']['home_search']["count"] < 42:
                    search_complete = True
                offset_parameter += 42

                for entry in json_data["data"]["home_search"]["results"]:
                    feature_dict = self.extract_features(entry)
                    feature_dict_list.append(feature_dict)

        return feature_dict_list

    def create_dataframe(self,zip) -> pd.DataFrame:
        feature_dict_list = self.parse_json_data(zip)
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
        df_new.drop(['id', 'house_type', 'subdivision', 'state', 'lat', 'lon', "city", "county"], axis=1,
                    inplace=True)
        df_new = df_new.dropna()
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
        return df_final


if __name__ == "__main__":
    r = RealtorZipScraper(page_numbers=10)
    df = r.create_dataframe('63043')
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

