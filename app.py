import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Animal Shelter Outcome Predictor",
    page_icon="🐾",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load("shelter_outcome_pipeline.pkl")


model = load_model()

st.title("🐾 Animal Shelter Outcome Prediction")
st.divider()

# ---------------- Inputs ----------------
animal_type = st.selectbox("Animal Type", ["Dog", "Cat"])
sex = st.selectbox("Sex", ["Male", "Female"])
spay_neuter = st.selectbox("Spay/Neuter", ["Yes", "No"])

age_group = st.selectbox(
    "Age Group",
    ["Puppy/Kitten", "Young", "Adult", "Senior"]
)

breed1 = st.text_input("Primary Breed", "")
breed2 = st.text_input("Secondary Breed", "")

color1 = st.text_input("Primary Color", "")
color2 = st.text_input("Secondary Color", "")

coat_pattern = st.text_input("Coat Pattern", "")
coat = st.text_input("Coat Type", "")

outcome_weekday = st.selectbox(
    "Outcome Weekday",
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday"]
)

periods = st.number_input("Periods", min_value=0, value=0)
period_range = st.number_input("Period Range", min_value=0, value=0)

birth_year = st.number_input(
    "Birth Year", min_value=2000, max_value=2035, value=2022)
birth_month = st.number_input(
    "Birth Month", min_value=1, max_value=12, value=1)

outcome_month = st.number_input(
    "Outcome Month", min_value=1, max_value=12, value=1)
outcome_year = st.number_input(
    "Outcome Year", min_value=2000, max_value=2035, value=2024)
outcome_hour = st.number_input(
    "Outcome Hour", min_value=0, max_value=23, value=12)

cfa_breed = st.checkbox("CFA Breed")
domestic_breed = st.checkbox("Domestic Breed")

# ---------------- Prediction ----------------
if st.button("Predict Outcome"):

    # Combined Features
    breed = f"{breed1}/{breed2}" if breed2 else breed1
    color = f"{color1}/{color2}" if color2 else color1
    age_text = age_group
    sex_status = f"{sex} {spay_neuter}"

    outcome_datetime = f"{int(outcome_year)}-{int(outcome_month):02d}-01 {int(outcome_hour):02d}:00:00"

    input_data = {
        "age_text": age_text,
        "animal_type": animal_type,
        "breed": breed,
        "color": color,
        "outcome_datetime": outcome_datetime,
        "outcome_subtype": "None",
        "sex_status": sex_status,
        "sex": sex,
        "spay_neuter": spay_neuter,
        "periods": float(periods),
        "period_range": float(period_range),
        "age_group": age_group,
        "birth_year": float(birth_year),
        "birth_month": float(birth_month),
        "outcome_month": float(outcome_month),
        "outcome_year": float(outcome_year),
        "outcome_weekday": outcome_weekday,
        "outcome_hour": float(outcome_hour),
        "breed1": breed1,
        "breed2": breed2,
        "cfa_breed": int(cfa_breed),
        "domestic_breed": int(domestic_breed),
        "coat_pattern": coat_pattern,
        "color1": color1,
        "color2": color2,
        "coat": coat
    }

    df = pd.DataFrame([input_data])

    # Replace blanks safely
    df.replace("", "Unknown", inplace=True)

    try:
        prediction = model.predict(df)[0]

        st.success("Prediction Completed ✅")
        st.subheader(f"Predicted Outcome: {prediction}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            confidence = round(max(proba[0]) * 100, 2)
            st.info(f"Confidence Score: {confidence}%")

    except Exception as e:
        st.error(f"Error: {e}")


print(os.getcwd())
