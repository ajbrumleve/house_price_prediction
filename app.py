import pickle

import streamlit as st

from pipeline import predict_specific_address


def main():
    # take input from users using st.form function
    with st.form("Address Search"):
        st.subheader("Please enter the following inputs:")
        # define variable to store user inputs
        zip_code = st.text_input("Zip code of the house:")
        house_number = st.text_input("House number of the house:")

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
    model_filename = 'MO_realtor_model.sav'
    df_filename = 'RealtorObject.sav'
    regr_model = pickle.load(open(model_filename, 'rb'))
    r = pickle.load(open(df_filename, 'rb'))
    main()
