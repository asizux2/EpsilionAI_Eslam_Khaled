import streamlit as st
from MEDA import load_data, clean_data,create_pie_chart,plot_histograms,plot_histograms_with_customizations,plot_correlation_heatmap


FILE_PATH = r"C:\Users\Lenovo\Desktop\Eslam_Final_Project\Sourse\online_course_engagement_data.csv"

# Load and clean the data
df = load_data(FILE_PATH)  
df = clean_data(df)   

# Title of the app
st.title('Model Evaluation and Data Analysis Dashboard')

# Pie Chart for Distribution of 'CourseCompletion'
st.header('CourseCompletion Distribution')
is_claim_count = df['CourseCompletion'].value_counts()
st.plotly_chart(create_pie_chart(df, 'CourseCompletion', 0))

# Define the numbers
completed_courses = 3568
not_completed_courses = 4641
total_courses = completed_courses + not_completed_courses
# Calculate percentages
completed_percentage = (completed_courses / total_courses) * 100
not_completed_percentage = (not_completed_courses / total_courses) * 100
# Streamlit app
st.title('Course Completion Analysis')
st.markdown('## Completion Rates')
st.write(f'- **{not_completed_courses:,}** courses have not been completed.')
st.write(f'- **{completed_courses:,}** courses have been completed.')
st.write(f'Approximately **{not_completed_percentage:.2f}%** of courses are not completed, while **{completed_percentage:.2f}%** reach completion.')
st.markdown("""
This balanced distribution suggests a need for further analysis to understand factors influencing course completion 
and to enhance overall course engagement strategies.
""")

# Pie Chart for Distribution of 'CourseCompletion'
st.header('CourseCategory Distribution')
is_claim_count = df['CourseCategory'].value_counts()
st.plotly_chart(create_pie_chart(df, 'CourseCategory', .5))
# Define the number of courses in each category
business_courses = 1671
programming_courses = 1655
science_courses = 1654
health_courses = 1648
arts_courses = 1581

# Streamlit app
st.title('Course Category Analysis')
st.markdown('## Course Distribution by Category')
# Display the number of courses in each category
st.write(f'- **Business Courses:** {business_courses} courses')
st.write(f'- **Programming Courses:** {programming_courses} courses')
st.write(f'- **Science Courses:** {science_courses} courses')
st.write(f'- **Health Courses:** {health_courses} courses')
st.write(f'- **Arts Courses:** {arts_courses} courses')
# Insights
st.markdown("""
Overall, the dataset shows a high number of courses across all categories, 
with Business, Programming, and Science having the highest counts.
""")
# Additional insights can be included if necessary
st.markdown("""
### Summary:
1. **Business Courses:**
   - There are **1,671** courses in the Business category, making it the most abundant category in the dataset.

2. **Programming Courses:**
   - With **1,655** courses, the Programming category is slightly behind Business but still holds a significant portion of the dataset.

3. **Science Courses:**
   - The Science category contains **1,654** courses, closely following Programming and Business in terms of quantity.

4. **Health Courses:**
   - There are **1,648** courses in the Health category, slightly fewer than those in Science.

5. **Arts Courses:**
   - The Arts category has the fewest courses with **1,581**, though it remains a notable category within the dataset.
""")


# Histograms for numerical columns
st.header('Histograms of Numerical Columns')
cols = ['TimeSpentOnCourse', 'NumberOfVideosWatched', 'CompletionRate', 'NumberOfQuizzesTaken', 'QuizScores']
st.plotly_chart(plot_histograms(df, cols))

# Display insights
st.header("1. Course Category Distribution")
st.write(
    """
    - The most frequent course categories, ranked from highest to lowest, are:
        1. **Business**
        2. **Programming**
        3. **Science**
        4. **Health**
        5. **Arts**
    """
)

st.header("2. Time Spent on Courses")
st.write(
    """
    - The majority of users spent between **49 to 51 hours** on courses.
    - In contrast, the least amount of time was spent by users in the range of **99 to 101 hours**.
    """
)

st.header("3. Number of Videos Watched")
st.write(
    """
    - Courses with an average of **10 videos watched** are the most common.
    """
)

st.header("4. Number of Quizzes Taken")
st.write(
    """
    - Most courses have a consistent average number of quizzes taken.
    """
)

st.header("5. Quiz Scores")
st.write(
    """
    - The highest recorded quiz score is **75**.
    """
)


# Histograms for columns with subplots compared to 'is_claim'
st.header("Histograms Showing Distribution of Course Metrics by Completion Status")
# Radio button for selecting one column
all_columns = ['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken', 'QuizScores', 'CompletionRate', 'CourseCategory']
selected_column = st.radio("Choose a column to display:", all_columns)

st.plotly_chart(plot_histograms_with_customizations(df, selected_column,'CourseCompletion'))

st.markdown("""
1. **Course Completion by Time Spent:**
   - Courses with less than 20 hours of time spent have a completion rate of less than 25%.
   - Courses with more than 70 hours of time spent have a balanced completion and non-completion rate, indicating that both completion and non-completion rates are similar.

2. **Course Completion by Number of Videos Watched:**
   - Courses where fewer than 5 videos are watched have a higher percentage of non-completion.
   - Conversely, courses with more than 6 videos watched show a lower percentage of non-completion, suggesting that more videos correlate with higher completion rates.

3. **Impact of Number of Quizzes on Completion Rate:**
   - As the number of quizzes increases, the percentage of course completions rises, while the percentage of non-completions decreases. This indicates a positive correlation between the number of quizzes and course completion rates.
""")

st.header('Correlation Heatmap')
st.plotly_chart(plot_correlation_heatmap(df))
