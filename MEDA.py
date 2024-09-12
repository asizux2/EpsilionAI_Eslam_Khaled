import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# function to load data
def load_data(file_path):                     
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

# Data Cleaning Function
def clean_data(df):
    """Clean the dataset by handling missing values and correcting data types."""
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    
    # Drop duplicates and reset the index
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Convert data types
    df['TimeSpentOnCourse'] = pd.to_numeric(df['TimeSpentOnCourse'], errors='coerce').astype('float')
    df['NumberOfVideosWatched'] = pd.to_numeric(df['NumberOfVideosWatched'], errors='coerce').round().astype('Int64')
    df['QuizScores'] = pd.to_numeric(df['QuizScores'], errors='coerce').astype('float')
    
    # Handle missing values in numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    return df


# function to create pie chart 
def create_pie_chart(df, column,r):
    """Create a pie chart for a given column."""
    return px.pie(df, names=column, template='plotly_dark', hole=r)


def plot_histograms(df, columns, num_cols=3, height=800, width=1000):
    # Number of columns to plot
    num_columns = len(columns)
    
    # Determine the number of rows needed for the subplots grid
    num_rows = (num_columns + num_cols - 1) // num_cols
    
    # Create a subplot figure
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=columns)
    
    # Loop through each column and add a histogram to the subplot grid
    for i, col in enumerate(columns):
        row = i // num_cols + 1  # Calculate the row index
        col_pos = i % num_cols + 1  # Calculate the column index
        
        # Add histogram for each column
        fig.add_trace(go.Histogram(x=df[col], name=col), row=row, col=col_pos)
    
    # Update layout for the entire figure
    fig.update_layout(height=height, width=width, showlegend=True, title_text='Histograms of DataFrame Columns')
    
    # Adjust spacing between subplots
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)
    
    # Display the plot
    return fig


def plot_histograms_with_customizations(df, col, color_column, marginal='box', barmode='group', template='plotly_dark'):
    fig = px.histogram(
        df, 
        x=col, 
        barmode=barmode, 
        color=color_column, 
        marginal=marginal, 
        template=template
    )
    return fig   


def plot_correlation_heatmap(df, drop_columns=None, color_scale='Viridis', height=600, width=800, title='Correlation Heatmap'):
    if drop_columns:
        df = df.drop(columns=drop_columns)
    
    # Calculate the correlation matrix
    correlation_matrix = df.corr(numeric_only=True)
    
    # Create the heatmap using Plotly Express
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,  # Automatically show values in the heatmap cells
        color_continuous_scale=color_scale,  # Use specified color scale
        labels={'color': 'Correlation'},  # Label for the color bar
        title=title  # Title for the heatmap
    )
    
    # Update layout to improve appearance
    fig.update_layout(
        height=height,  # Set the height of the figure
        width=width,  # Set the width of the figure
        xaxis_title='Variables',  # Label for the x-axis
        yaxis_title='Variables',  # Label for the y-axis
        title_x=0.5  # Center the title
    )
    
    # Display the heatmap
    return fig        