import sys

import pandas as pd


class Zips:
    def __init__(self, list_county, state_abbr):
        """
                Initialize a Zips object with the specified county/ies and state abbreviation.

                Args:
                - list_county (list): A list of county/ies to retrieve zip codes from. If empty, all zip codes for the specified state (excluding "HH") will be retrieved.
                - state_abbr (str): The state abbreviation to filter zip codes.

                Attributes:
                - list_county (list): The list of county/ies specified.
                - state_abbr (str): The state abbreviation specified.
                - list_zips (list): The list of zip codes retrieved.

        """
        self.list_county = list_county
        self.state_abbr = state_abbr
        self.list_zips = []

    def get_zip_list(self):
        """
           Retrieve a list of zip codes based on the specified state abbreviation and county/ies.

           If self.list_county is empty, it retrieves all zip codes for the specified state, excluding those containing "HH".
           If self.list_county contains county/ies, it retrieves zip codes for the specified county/ies in the specified state, excluding those containing "HH".

           Returns:
           - total_list (list): A sorted list of zip codes
        """
        # Initialize an empty list to store the zip codes
        total_list = []
        # Load the zip codes from the CSV file
        zips = pd.read_csv('files/zip_codes.csv')
        # If no county is specified
        if len(self.list_county) == 0:
            # Filter zip codes based on state and excluding "HH"
            zips_list = zips[(zips["state_abbr"] == self.state_abbr) & (
                    zips['zipcode'].str.contains("HH") == False)]["zipcode"]
            # Add the filtered zip codes to the total list
            total_list.extend(zips_list)
            # Sort the zip codes in ascending order
            total_list.sort()
            # Update the instance variable with the final zip code list
            self.list_zips = total_list
        # If county/ies are specified
        else:
            # If no county is specified, filter zip codes based on state, sort the zip codes in ascending order,
            # and update the instance variable with the final zip code list
            for county in self.list_county:
                zips_list = zips[(zips["county"] == county) & (zips["state_abbr"] == self.state_abbr) & (
                            zips['zipcode'].str.contains("HH") == False)]["zipcode"]
                total_list.extend(zips_list)
            total_list.sort()
            self.list_zips = total_list
        # Return the sorted list of zip codes
        return total_list

    def get_county_list(self):
        zips = pd.read_csv('files/zip_codes.csv')
        zips_f = zips[zips["state_abbr"] == self.state_abbr]
        county_series = zips_f['county']
        county_choices = list(set(county_series))
        county_choices.sort()
        return county_choices
