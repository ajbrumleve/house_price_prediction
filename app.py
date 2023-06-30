import os
import pickle

import streamlit as st

from pipeline import predict_specific_address, find_deals
from config import STATE_ABBR
from zips import Zips


# Function to load the regression model and data based on the selected state
def load_data(state):
    try:
        model_filename = f'models/{state}_realtor_model.sav'
        df_filename = f'models/{state}_RealtorObject.sav'
        regression_model = pickle.load(open(model_filename, 'rb'))
        real_obj = pickle.load(open(df_filename, 'rb'))
        return real_obj, regression_model
    except:
        if state == "":
            pass
        else:
            st.write(f"Model not available for {state}")


# Function to get user inputs for state and activity choice
def get_user_inputs():
    st.subheader("Models are trained for each state. What state are you interested in?")
    state = st.selectbox("State Abbreviation", STATE_ABBR)

    st.subheader("Choose an activity:")
    choice = st.radio("Select an option:", ["Look up a house", "See filtered table of all houses"])
    if st.button("Submit"):
        return state, choice
    return "", "Look up a house"


# Function to predict the price for a specific address
def predict_price(real_obj, regression_model):
    st.subheader("To compare the actual and predicted price of a house, please enter the following inputs:")
    possible_zips = real_obj.zips_df
    addresses = real_obj.address_df["address"]
    zip_code = st.selectbox("Zip code of the house:", possible_zips)
    house_number = st.selectbox("House number of the house:", addresses)

    submit = st.button("Predict")

    if submit:
        address_price = predict_specific_address(real_obj, regression_model, zip_code, house_number)
        if isinstance(address_price, str):
            st.write(address_price)
        elif address_price[0] > address_price[1]:
            st.write(
                f"The model predicts a price of ${address_price[1]}. The actual price is ${address_price[0]}. The house is ${address_price[0] - address_price[1]} more expensive than the prediction.")
        elif address_price[0] < address_price[1]:
            st.write(
                f"The model predicts a price of ${int(address_price[1])}. The actual price is ${int(address_price[0])}. The house is ${int(address_price[1]) - int(address_price[0])} cheaper than the prediction.")
        elif address_price[0] == address_price[1]:
            st.write(f"The model predicts the exact price of ${address_price[0]}")
        else:
            st.write("Address not found")


def get_filter_inputs(r, model, state):
    st.subheader("Enter the filter criteria:")
    county_choices = Zips([],state).get_county_list()
    min_beds = st.number_input("Minimum Bedrooms", min_value=0, step=1, value=0)
    min_sqft = st.number_input("Minimum Square Footage", min_value=0, step=1, value=0)
    max_price = st.number_input("Maximum Price", min_value=0, step=1, value=0)
    counties = st.multiselect("Counties", county_choices)

    submit = st.button("Search")

    if submit:
        # Call the find_deals() function with the user inputs
        filtered_df = find_deals(r, model, min_beds, min_sqft, max_price, counties, state)
        # Display the filtered table of houses
        st.dataframe(filtered_df)


def main():
    state, choice = get_user_inputs()
    try:
        real_obj, regression_model = load_data(state)

        if choice == "Look up a house":
            try:
                predict_price(real_obj, regression_model)
            except:
                pass
        elif choice == "See filtered table of all houses":
            get_filter_inputs(real_obj, regression_model, state)
    except:
        pass


if __name__ == '__main__':
    main()
