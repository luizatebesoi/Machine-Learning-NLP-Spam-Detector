# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#%matplotlib inline

# Loading the data
messages = pd.read_csv('SMSSpamCollection', sep = '\t', names = ['label', 'message'])

# Displaying aggregated information about the data
messages.head()
messages.describe()
print('\n')
messages.info()
print('\n')
messages.groupby('label').describe()

# Creating a new column 'length' to display the length of each message
messages['length'] = messages['message'].apply(len)

# Exploratory Data Analysis

# Creating a histogram showing the relationship between the length of the messages and their label. It can be noticed that Spam messages tend to have bigger lengths.
messages.hist(column= 'length', by= 'label', bins = 60, figsize= (12,4))

# Getting the data ready for the model

# Creating a text processing function to remove punctuation and common words
def text_process(mess):
    """
    1. remove punctuation
    2. remove stopwords
    3. return list of cleaned words
    """
    mess = ''.join([char for char in mess if char not in string.punctuation])
    words = [word for word in mess.split() if word.lower() not in stopwords.words('english')]
    return words

# Train Test Split the data
X = messages['message']
y = messages['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Building and training the classifier
pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer = text_process)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
        ])
    
pipeline.fit(X_train, y_train)

# Predicting the results
y_pred = pipeline.predict(X_test)

# Evaluating the model
print(classification_report(y_test, y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))





