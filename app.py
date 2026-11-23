import streamlit as st
import pandas as pd
import joblib


# Load model and threshold
model = joblib.load("Desktop/coustmer_intension/archive/purchase_intent_model.pkl")
threshold = joblib.load("Desktop/coustmer_intension/archive/threshold.pkl")


st.title("🛒 Purchase Intent Predictor")

st.write(
    "Enter the user's session behavior to predict purchase intent."
)


# User inputs
views = st.number_input(
    "Number of Views",
    min_value=0,
    value=1
)

add_to_cart = st.number_input(
    "Add to Cart",
    min_value=0,
    value=0
)

unique_item_viewed = st.number_input(
    "Unique Items Viewed",
    min_value=0,
    value=1
)

session_duration_min = st.number_input(
    "Session Duration (minutes)",
    min_value=0.0,
    value=0.0
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=7
)

day = st.number_input(
    "Day of Week",
    min_value=0,
    max_value=6,
    value=3
)

hour = st.number_input(
    "Hour",
    min_value=0,
    max_value=23,
    value=12
)


# Prediction button
if st.button("Predict Purchase Intent"):

    input_data = pd.DataFrame([{
        "views": views,
        "add_to_cart": add_to_cart,
        "unique_item_viewed": unique_item_viewed,
        "session_duration_min": session_duration_min,
        "month": month,
        "day": day,
        "hour": hour
    }])

    # Purchase probability
    probability = model.predict_proba(input_data)[0, 1]

    # Apply threshold
    prediction = int(probability >= threshold)

    st.subheader("Prediction")

    st.write(
        f"Purchase Probability: **{probability:.2%}**"
    )

    if prediction == 1:
        st.success("🟢 Likely to Purchase")
    else:
        st.info("🔵 Unlikely to Purchase")