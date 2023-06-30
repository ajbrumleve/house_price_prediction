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
        submit = st.form_submit_button("Submit")

    if submit:
        filename = f'models/{state}_realtor_model.sav'
        if os.path.exists(filename):
            model_filename = f'models/{state}_realtor_model.sav'
            df_filename = f'models/{state}_RealtorObject.sav'
            regression_model = pickle.load(open(model_filename, 'rb'))
            real_obj = pickle.load(open(df_filename, 'rb'))
            st.session_state['data_1'] = {"r":real_obj, "model":regression_model, "state":state}
            st.session_state['section'] = 'section 2'
            return real_obj, regression_model, state,"confirmed"

        else:
            st.write("File does not exist.")
def main(r, regr_model, state_abbr):
    # take input from users using st.form function
    with st.form("Address Search"):
        st.subheader("To compare the actual and predicted price of a house, please enter the following inputs:")
        # define variable to store user inputs
        possible_zips = r.zips_df
        addresses = r.address_df["address"]
        print(possible_zips[:10])
        print(addresses[:10])
        zip_code = st.selectbox("Zip code of the house:",possible_zips)
        house_number = st.selectbox("House number of the house:",addresses)

        # put a submit button to predict the output of the model
        submit2 = st.form_submit_button("Predict")

    if submit2:
        address_price = predict_specific_address(r, regr_model, zip_code, house_number)
        if isinstance(address_price, str):
            st.write(address_price)
        elif address_price[0] > address_price[1]:
            st.write(f"The model predicts a price of \${address_price[1]}. The actual price is \${address_price[0]}. The house is \${address_price[0] - address_price[1]} more expensive than the prediction.")
        elif address_price[0] < address_price[1]:
            st.write(f"The model predicts a price of \${int(address_price[1])}. The actual price is \${int(address_price[0])}. The house is \${int(address_price[1]) - int(address_price[0])} cheaper than the prediction.")
        elif address_price[0] == address_price[1]:
            st.write(f"The model predicts the exact price of \${address_price[0]}")

def make_choice(r, regr_model, state_abbr):
    st.subheader("Choose an App:")
    choice = st.radio("Select an option:", ["Look up a house", "See filtered table of all houses"])
    submit3 = st.button("Submit")

    if submit3:
        if choice == "Look up a house":
            # Logic for looking up a specific house
            st.write("Look up house")
            # main(r, regr_model, state_abbr)

        else:
            # Logic for showing a filtered table of all houses
            st.write("You selected: See filtered table of all houses")
    return "","","",""

if __name__ == '__main__':
    if 'section' not in st.session_state:
        st.session_state['section'] = 'section 1'
    if st.session_state['section'] == 'section 1':
        confirm_state()
    if st.session_state['section'] == 'section 2:
        make_choice()
    if status == "confirmed":
        make_choice(real_obj, regression_model, state)
    # main(r, regr_model, state_abbr)

