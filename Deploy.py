import streamlit as st
import pandas as pd

import joblib
import altair as alt
from pandas import read_csv

# Load the trained models
linear_model = joblib.load('linear_regression_model.joblib')
random_forest_model = joblib.load('random_forest_model.joblib')
decision_tree_model = joblib.load('decision_tree_model.joblib')
scaler = joblib.load('scaler.joblib')  # Load the scaler used during training

df = read_csv('Data_Long.csv')

st.title("""
Anomaly Global Temperature - 1880 - 20224 Line Graph
""")
chart = alt.Chart(df).mark_line().encode(x= alt.X('Year', title='Years 1880 - 2024'),
                                          y = alt.Y("5Year_Avg", title='Temperature Value (Standardized)'))

st.altair_chart(chart, use_container_width=True)

st.write("Click on the button to find out more in Tableau!")
st.link_button("Tableau Dashboard", "https://public.tableau.com/views/Tableau_OESON/Dashboard2?:language=en-GB&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link")

# Title and description
st.title("Global Anomaly Temperature Predictor")

st.write("""
This application predicts the annual anomaly temperature based on the 4 seasonal temperatures.
Choose the model you want to use.
""")

# Sidebar for selecting the model
model_choice = st.sidebar.selectbox(
    "Select a Model",
    ("Linear Regression", "Decision Tree", "Random Forest")
)

# Input fields for user data
st.header("Enter Input Data")
SUMMER = st.text_input("SUMMER")
AUTUMN = st.text_input("AUTUMN")
WINTER = st.text_input("WINTER")
SPRING = st.text_input("SPRING")


# Prediction
if st.button("Predict Annual Temperature "):
    # Organize input data into a dataframe
    input_data = pd.DataFrame({
        'SUMMER' : [SUMMER],
        'AUTUMN' : [AUTUMN],
        'WINTER' : [WINTER],
        'SPRING' : [SPRING],
    })

    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Choose model for prediction
    if model_choice == "Linear Regression":
        prediction = linear_model.predict(scaled_data)[0]

    elif model_choice == "Decision Tree":
        prediction = decision_tree_model.predict(scaled_data)[0]

    elif model_choice == "Random Forest":
        prediction = random_forest_model.predict(scaled_data)[0]

    st.success(f"The predicted Annual Temperature is: {prediction:.2f}")
