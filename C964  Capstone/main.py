#Modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import ipywidgets as widgets
from IPython.display import display
#Load Data
url = 'https://raw.githubusercontent.com/vlohana103/C964-Capstone/refs/heads/main/netflix_titles.csv' #data set from kaggle sent to github in raw format (CSV)
df = pd.read_csv(url) #Assigns the variable df to the url enclosed in pandas built in read function

#Bar Chart to compare Movies and TV Shows
sns.countplot (data=df, x ='type')
plt.title('Number of Movies vs TV Shows')
plt.xlabel('Media Type')
plt.ylabel('Amount')
plt.show()

#Histogram to show release year of Moives and TV Shows
sns.histplot(data=df, x='release_year', bins=10)
plt.title('Titles by Year Released')
plt.xlabel('Year Released')
plt.ylabel('Titles')
plt.show()

#Pie Chart to show the amound of different rating of Movies and TV Shows
rating_counts = df['rating'].value_counts().head(7)
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%')
plt.title('Top 7 Media Ratings')
plt.show()

#Convert Data

# Gets number of minutes for Movie time
def get_min(time):
    if 'min' in str(time):#Movie
        return int(str(time).split()[0]) # splits the first index which is a number to check if it's a movie
    else:
        return 0

#Gets number of Season for TV Show
def get_season(time):
    if "Season" in str(time): #TV Show
        return int(str(time).split()[0])
    else:
        return 0

#Creates duration_min variable which hiolds the contents from duration for Movies from csv
df['duration_min'] = df['duration'].apply(get_min)

#Creates duration_season variable which holds the contents from the duration for TV Shows from csv
df['duration_season'] = df['duration'].apply(get_season)

df['type'] = df['type'].map({'Movie': 0, 'TV Show': 1}) #Uses type in CSV and assigns 0 to movie, and 1 for tv show
df['rating_name'] = df['rating'] #stores rating df into rating_name variable to implement in UI
df['rating_code'] = df['rating'].astype('category').cat.codes # Uses rating in csv and assigns it to rating variable, assigns all movie and show ratings to numbers.

#Implement ML
x = df[['rating_code','duration_min', 'duration_season']] #splits features into a dataframe as input
y = df['type'] # target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #splits data in to 80% training, 20% testing, and keeps the split the same

model = LogisticRegression() #creates Logistic Regression Model

model.fit(x_train, y_train) #trains the model with the split training parameters

y_predict = model.predict(x_test) #predicts from x_test

#Measure and show accuracy
print("Accuracy: ", accuracy_score(y_test, y_predict)) #measures accuracy by comparing test to predict and prints it out
print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict)) # Creates confusion matrix using y_test and y_predict to show Logistic Regression Model's performance

#UI
#duration slider:
duration_widget = widgets.IntSlider(
    value=20, #slider default value
    min=1, #in minutes
    max=250, #250 in minutes
    step = 1,
    description='Runtime in Minutes: '
)


#seasons widget
season_widget = widgets.IntSlider(
    value = 0,
    min =0,
    max=10,
    step=1,
    description='Seasons: '
)


#rating dropdown:
rating_widget = widgets.Dropdown(
    options=df['rating_name'].astype('category').cat.categories,
    description='Rating'
)


#output button:
predict_button = widgets.Button(description="Predict") #button name
output = widgets.Output() #shows results

#button has been pushed
def button_clicked(push):
        with output:
            output.clear_output() #removes old output information
            duration = duration_widget.value #duration slider value
            seasons = season_widget.value #seasons slider value
            rating_name = rating_widget.value #uses ratings drop down
            rating_code = df['rating_name'].astype('category').cat.categories.get_loc(rating_name) #changes rating from dropdown into numbers
            input_df = pd.DataFrame([[rating_code, duration, seasons]],columns=['rating_code', 'duration_min', 'duration_season']) #creates df for input
            prediction = model.predict(input_df) #model is implemented here to predict from input
            print("Prediction (0=Movie, 1=TV Show): ", prediction[0]) #prints predicition

#Buttons implemented with functions
predict_button.on_click(button_clicked) #functionality for the predict button

#Display widgets
display(duration_widget, season_widget, rating_widget, predict_button, output) #displays all widgets
