import pickle

import streamlit as st

import config
from pipeline import predict_specific_address, find_deals
from zips import Zips

# Define the available states
states = config.STATE_ABBR

# Define the available activities
activities = ["Look up a house", "Create filtered dataframe"]

# Function to handle "Look up a house" activity
def look_up_house():
    st.subheader("To compare the actual and predicted price of a house, please enter the following inputs:")
    real_obj = st.session_state['real_obj']
    regression_model = st.session_state['regression_model']
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
                f"The model predicts a price of \${address_price[1]}. The actual price is \${address_price[0]}. The house is \${address_price[0] - address_price[1]} more expensive than the prediction.")
        elif address_price[0] < address_price[1]:
            st.write(
                f"The model predicts a price of \${int(address_price[1])}. The actual price is \${int(address_price[0])}. The house is \${int(address_price[1]) - int(address_price[0])} cheaper than the prediction.")
        elif address_price[0] == address_price[1]:
            st.write(f"The model predicts the exact price of \${address_price[0]}")
        else:
            st.write("Address not found")

# Function to handle "Create filtered dataframe" activity
def create_filtered_dataframe():
    st.subheader("Create filtered dataframe")
    # Get user inputs (e.g., filters)
    min_bedrooms = st.slider("Minimum bedrooms", 1, 5, 1)
    min_sqft = st.slider("Minimum square footage", 100, 2000, 100)
    max_price = st.slider("Maximum price", 100000, 1000000, 100000)
    counties = st.multiselect("Counties", st.session_state['county_choices'])
    real_obj = st.session_state['real_obj']
    regression_model = st.session_state['regression_model']
    submit = st.button("Search")

    if submit:
        # Call the find_deals() function with the user inputs
        filtered_df = find_deals(real_obj, regression_model, min_bedrooms, min_sqft, max_price, counties, st.session_state['state'])
        selected_columns = ['address', 'city', 'county','postal_code','beds','baths','sqft','lot_sqft','price','prediction','price_diff']

        # Display the filtered table of houses
        st.dataframe(filtered_df[selected_columns],hide_index=True)

# Main program
def main():
    # Initialize session state
    if "state" not in st.session_state:
        st.session_state["state"] = None

    # Get user state input
    state = st.selectbox("Select your state", states)

    if st.button("Submit State"):
        st.session_state["state"] = state

    # Check the current state
    current_state = st.session_state["state"]


    # Display appropriate activity based on the state
    if current_state is None:
        st.write("Please select a state.")
    else:
        model_filename = f'models/{current_state}_realtor_model.sav'
        df_filename = f'models/{current_state}_RealtorObject.sav'
        st.session_state['regression_model'] = pickle.load(open(model_filename, 'rb'))
        st.session_state['real_obj'] = pickle.load(open(df_filename, 'rb'))
        st.session_state['county_choices'] = Zips([], current_state).get_county_list()
        activity = st.selectbox("Select an activity", activities)
        # if st.button("Submit Activity"):
        st.session_state["activity"] = activity
        current_activity = st.session_state.get("activity")
        if current_activity == "Look up a house":
            look_up_house()
        elif current_activity == "Create filtered dataframe":
            create_filtered_dataframe()


# Run the main program
if __name__ == "__main__":
    main()
