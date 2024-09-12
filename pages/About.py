import streamlit as st

# Title of the App
st.title("Online Course Completion Prediction")

# Dataset Overview Section
st.header("Dataset Overview")
st.write("""  
The dataset contains information on students enrolled in online courses, including features such as course category, 
time spent on the course, number of videos watched, quizzes taken, quiz scores, completion rate, and the device type used. 
The target variable, `CourseCompletion`, indicates whether a student is likely to complete the course or not.
""")

# Data Cleaning Section
st.header("Data Cleaning")
st.write("""
- **Initial Size**: 9180 records, 9 features.
- **After Cleaning**: 8209 records, 9 features (after removing duplicates and irrelevant data).
""")


# Analysis Focus Section
st.header("Analysis Focus")
st.write("""  
- **Target Variable**: The analysis primarily focused on `CourseCompletion` to understand factors influencing the likelihood of completing the course.
- **Distribution Analysis**: Examined the distribution of `CourseCompletion` and its relationship with key features.
""")
# Machine Learning Process Section
st.header("Machine Learning Process")
st.write("""  
- **Encoding**: Categorical features were transformed using one-hot encoding.
- **Scaling**: Continuous features were scaled using `StandardScaler`.
- **Modeling**: Built classification models to predict `CourseCompletion`, including tuning hyperparameters and cross-validation.
""")


# Conclusion Section
st.header("Conclusion")
st.write("""  
The process led to the development of a predictive model that helps identify students likely to complete or not complete the course, 
enabling targeted interventions and enhancing course management strategies.
""")
