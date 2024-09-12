import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Title of the app
st.title("Course Metrics Prediction")

# Try to load the pre-trained model and preprocessor
try:
    model = joblib.load("Sourse/best_random_forest_model.joblib")
    preprocessor = joblib.load("Sourse/preprocessor.pkl")  # Ensure .pkl is the correct extension
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Please check the file path.")
    st.stop()

# Input fields for user data
course_category = st.selectbox("Course Category", ['Health', 'Arts', 'Science', 'Programming', 'Business'])
time_spent_on_course = st.number_input("Time Spent on Course (in hours)", min_value=0.0, format="%.2f")
number_of_videos_watched = st.number_input("Number of Videos Watched", min_value=0)
number_of_quizzes_taken = st.selectbox("Number of Quizzes Taken", list(range(11)))
quiz_scores = st.number_input("Quiz Scores", min_value=0.0, format="%.2f")
completion_rate = st.number_input("Completion Rate", min_value=0.0, format="%.2f")
device_type = st.radio("Device Type", ["Desktop", "Mobile"])

# Convert device type to numeric values
device_type_numeric = 0 if device_type == "Desktop" else 1

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'CourseCategory': [course_category],
    'TimeSpentOnCourse': [time_spent_on_course],
    'NumberOfVideosWatched': [number_of_videos_watched],
    'NumberOfQuizzesTaken': [number_of_quizzes_taken],
    'QuizScores': [quiz_scores],
    'CompletionRate': [completion_rate],
    'DeviceType': [device_type_numeric]
})

# Preprocess the input data
try:
    input_data_processed = preprocessor.transform(input_data)
except Exception as e:
    st.error(f"Error during data preprocessing: {e}")
    st.stop()

# Prediction button
if st.button('Predict'):
    try:
        # Make prediction
        prediction = model.predict(input_data_processed)
        prediction_proba = model.predict_proba(input_data_processed)

        # Display result
        st.write("Prediction:", "Completed" if prediction[0] == 1 else "Not Completed")

        if prediction[0] == 1:
            st.success('Student Will Complete the Course.')
        elif prediction[0] == 0:
            st.error('Student Will Not Complete the Course.')

        # Display prediction probabilities
        st.subheader('Prediction Probability')
        st.write(f"**Probability of Not Completing:** {prediction_proba[0][0]:.2%}")
        st.write(f"**Probability of Completing:** {prediction_proba[0][1]:.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")


