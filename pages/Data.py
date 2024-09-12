# Import necessary libraries
import pandas as pd 
import streamlit as st 

# Title for the page
st.markdown("<center><h1> Online Courses Dataset </h1></center>", unsafe_allow_html=True)

# Displaying information about the dataset before cleaning and preprocessing
st.write('Online Courses Dataset Before Cleaning And Preprocessing:')

# Reading and displaying the dataset before cleaning and preprocessing
df_before = pd.read_csv(r"C:\Users\Lenovo\Desktop\Eslam_Final_Project\Sourse\online_course_engagement_data.csv")
st.write(df_before)

# Displaying information about the dataset after cleaning and preprocessing
st.write('Online Courses Dataset After Cleaning And Preprocessing:')

# Reading and displaying the cleaned dataset
df_after = pd.read_csv(r"C:\Users\Lenovo\Desktop\Eslam_Final_Project\Sourse\cleaned_data.csv")
st.write(df_after)
