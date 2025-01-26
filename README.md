# Hate-Speech-Detection
## Overview
This project aims to build a model to detect hate speech in tweets. The dataset used contains tweets labeled as hate speech, offensive language, or neither. The project involves cleaning the data, transforming it using various text processing techniques, and training a machine learning model to classify the tweets.

### Dataset
The dataset used in this project is a CSV file (labeled_data.csv) containing the following columns:

1. Unnamed: 0: Index
2. count: Count of different classes
3. hate_speech: Count of hate speech instances
4. offensive_language: Count of offensive language instances
5. neither: Count of instances that are neither hate speech nor offensive
6. class: Class label (0 for hate speech, 1 for offensive language, 2 for neither)
7. tweet: The tweet text

### Installation
To run this project, you'll need the following libraries:

1. pandas
2. numpy
3. nltk
4. scikit-learn
5. seaborn
6. matplotlib

## Data Preparation
### Loading the Data:
code:
import pandas as pd
dataset = pd.read_csv("labeled_data.csv")

### Exploring the Data:
code:
print(dataset.head())
print(dataset.info())
print(dataset.describe())

### Handling Missing Values:
code:
dataset.isnull().sum()

### Adding Labels:
code:
dataset["labels"] = dataset["class"].map({0: "Hate speech", 1: "Offensive language", 2: "No hate nor offensive"})
data = dataset[["tweet", "labels"]]

## Data Cleaning
### Text Cleaning Function:
code:
import re
import nltk
import string
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer('english')

def clean_data(text):
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean_data)

## Model Training
### Feature Extraction:
code:
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(data["tweet"])
y = data["labels"]

### Splitting the Data:
code:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

### Training the Model:
code:
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

## Model Evaluation
### Making Predictions:
code:
y_pred = dt.predict(x_test)

### Confusion Matrix:
code:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt=".1f", cmap="YlGnBu")
plt.show()

### Accuracy Score:
code:
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

## Sample Predictions
### Predicting New Samples:
code:
sample = "Let's unite and kill all the people who are protesting against the government"
sample = clean_data(sample)
data1 = cv.transform([sample]).toarray()
print(dt.predict(data1))

sample1 = "Yummy, I wanna eat you up"
sample1 = clean_data(sample1)
data2 = cv.transform([sample1]).toarray()
print(dt.predict(data2))

## Conclusion
This project demonstrates how to preprocess text data, train a machine learning model, and make predictions for detecting hate speech in tweets. The Decision Tree Classifier used here provides a baseline model which can be further improved with more advanced techniques and hyperparameter tuning.
