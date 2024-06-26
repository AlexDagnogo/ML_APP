import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

with open('classifier2.pkl', 'rb') as file:
    modele = pickle.load(file)

st.title("Prédiction des personnes susceptibles de posséder un compte bancaire.")

st.write("***L'ensemble de données contient des informations démographiques et les services financiers utilisés par environ 33 600 personnes en Afrique de l'Est.***")

col1, col2 = st.columns(2)
year = col1.number_input("ANNEE D'INTERROGGATION", min_value=2016, max_value=2018, value=2017)
gender_of_respondent = col2.number_input("GENRE :     0 - FEMME | 1 - HOMME", min_value=0, max_value=1, value=1)
location_type = col1.number_input("TYPE DU LIEU :     0 - RURAL | 1 - URBAN", min_value=0, max_value=1, value=0)
cellphone_access = col2.number_input("TELEPHONE :     0 - NO | 1 - YES", min_value=0, max_value=1, value=1)
household_size = col1.number_input("PERSONNES EN CHARGE", min_value=0, max_value=20, value=2)
age_of_respondent = col2.number_input("AGE DU CONCERNE", min_value=0, max_value=150, value=40)
marital_status	= st.number_input("ETAT CIVIL :  \n0 - Divorced/Seperated | 1 - Dont know | 2 - Married/Living together | 3 - Single/Never Married | 4 - Widowed", min_value=0, max_value=4, value=2)
education_level = st.number_input("NIVEAU D'EDUCATION :  \n0 - No formal education | 1 - Other/Dont know/RTA | 2 - Primary education | 3 - Secondary education | 4 - Tertiary education | 5 - Vocational/Specialised training", min_value=0, max_value=5, value=4)

input_data = np.array([[year, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, marital_status, education_level]])

if st.button('Prédire'):
    prediction = modele.predict(input_data)
    st.write(f'Prédiction : {"Le concerné possède un compte bancaire" if prediction[0] == 1 else "Le concerné ne possède pas un compte bancaire"}')