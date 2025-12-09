🔹 Project Overview

This project is a machine learning application that predicts whether a Netflix title is a Movie or a TV Show based on its rating, duration, and number of seasons.

The dataset is sourced from Kaggle (Netflix Titles Dataset

) and contains metadata about movies and TV shows available on Netflix.

This project demonstrates:

Data cleaning and feature engineering

Exploratory data analysis (EDA)

Machine learning model training using Logistic Regression

Interactive UI using ipywidgets for predictions

Visualization of distributions using Matplotlib and Seaborn

🔹 Key Features
1️⃣ Exploratory Data Analysis (EDA)

Bar chart: Compare number of Movies vs TV Shows

Histogram: Show number of titles released per year

Pie chart: Display top 7 media ratings

2️⃣ Data Preprocessing

Extracts duration in minutes for Movies

Extracts number of seasons for TV Shows

Converts categorical ratings into numeric codes for ML

3️⃣ Machine Learning

Model: Logistic Regression

Input features: rating_code, duration_min, duration_season

Target: type (0 = Movie, 1 = TV Show)

Train/Test split: 80% training / 20% testing

Accuracy and confusion matrix displayed

4️⃣ Interactive UI

Duration slider: Select runtime in minutes

Seasons slider: Select number of seasons

Rating dropdown: Select media rating

Predict button: Returns whether the input is predicted to be a Movie or TV Show

🔹 Libraries Used
# Data manipulation
pandas, numpy

# Visualization
matplotlib.pyplot, seaborn

# Machine Learning
scikit-learn: LogisticRegression, train_test_split, accuracy_score, confusion_matrix

# UI
ipywidgets, IPython.display

🔹 How to Run

Install Python 3 (Anaconda/Miniconda recommended)

Install dependencies:

pip install pandas matplotlib seaborn scikit-learn ipywidgets


Download Notebook & Dataset:

Notebook is hosted on GitHub

Dataset is loaded directly from the repository

Run the Notebook in Jupyter:

jupyter notebook Netflix_Capstone.ipynb


Use the interactive widgets to test predictions:

Adjust Duration (minutes), Seasons, and Rating

Click Predict to see the model’s output

🔹 Model Accuracy

Accuracy Score: ~X%

Confusion Matrix: Shows true positives, false positives, etc.

Note: Replace X% with the actual accuracy after running your notebook.

🔹 Project Workflow

Load and inspect the data

Visualize data for trends and distributions

Feature engineering for Movie runtime and TV Show seasons

Convert categorical ratings to numerical codes

Train Logistic Regression model

Create an interactive UI for user input

Display predictions and validate model performance

🔹 Future Improvements

Use a more complex model (Random Forest, XGBoost) for higher accuracy

Include genre, director, or country as features

Expand the UI to allow batch predictions

Deploy as a web app for interactive access
