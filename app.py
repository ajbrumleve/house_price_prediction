import os
import pickle

import streamlit as st

from pipeline import predict_specific_address
from config import STATE_ABBR
from zips import Zips


def confirm_state():
    with st.form("Get State"):
        st.subheader("Models are trained for each state. What state are you interested in?")
        # define variable to store user inputs
        state = st.selectbox("State Abbreviation", STATE_ABBR)

        # put a submit button to predict the output of the model
        submit = st.form_submit_button("Predict")
    if submit:
        filename = f'models/{state}_realtor_model.sav'
        if os.path.exists(filename):
            st.write("File exists!")
            model_filename = f'models/{state}_realtor_model.sav'
            df_filename = f'models/{state}_RealtorObject.sav'
            regression_model = pickle.load(open(model_filename, 'rb'))
            real_obj = pickle.load(open(df_filename, 'rb'))
            main()
        else:
            st.write("File does not exist.")
        return real_obj, regression_model, state
def main(r, regr_model, state_abbr):
    # take input from users using st.form function
    with st.form("Address Search"):
        st.subheader("To compare the actual and predicted price of a house, please enter the following inputs:")
        # define variable to store user inputs
        possible_zips = r.zips_df
        addresses = r.address_df["address"]
        zip_code = st.selectbox("Zip code of the house:",possible_zips)
        house_number = st.text_input("House number of the house:",addresses)

        # put a submit button to predict the output of the model
        submit = st.form_submit_button("Predict")

    if submit:
        address_price = predict_specific_address(r, regr_model, zip_code, house_number)
        if isinstance(address_price, str):
            st.write(address_price)
        elif address_price[0] > address_price[1]:
            st.write(f"The model predicts a price of \${address_price[1]}. The actual price is \${address_price[0]}. The house is \${address_price[0] - address_price[1]} more expensive than the prediction.")
        elif address_price[0] < address_price[1]:
            st.write(f"The model predicts a price of \${int(address_price[1])}. The actual price is \${int(address_price[0])}. The house is \${int(address_price[1]) - int(address_price[0])} cheaper than the prediction.")
        elif address_price[0] == address_price[1]:
            st.write(f"The model predicts the exact price of \${address_price[0]}")


if __name__ == '__main__':
    r, regr_model, state_abbr = confirm_state()
    main(r, regr_model, state_abbr)

