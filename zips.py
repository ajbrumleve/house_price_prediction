import sys

import pandas as pd


class Zips:
    def __init__(self, list_county, state_abbr):
        self.list_county = list_county
        self.state_abbr = state_abbr
        self.list_zips = []

    def get_zip_list(self):
        total_list = []
        zips = pd.read_csv('files/zip_codes.csv')
        if len(self.list_county) == 0:
            zips_list = zips[(zips["state_abbr"] == self.state_abbr) & (
                    zips['zipcode'].str.contains("HH") == False)]["zipcode"]
            total_list.extend(zips_list)
            total_list.sort()
            self.list_zips = total_list
        else:
            for county in self.list_county:
                zips_list = zips[(zips["county"] == county) & (zips["state_abbr"] == self.state_abbr) & (
                            zips['zipcode'].str.contains("HH") == False)]["zipcode"]
                total_list.extend(zips_list)
            total_list.sort()
            self.list_zips = total_list
        return total_list

