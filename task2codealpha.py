import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


# Update the path to your file location
file_path = 'C:\\Users\\YourUsername\\Desktop\\training.1600000.processed.noemoticon.csv'

# Load the dataset
data = pd.read_csv(file_path, encoding='latin-1', header=None)

# Rename the columns for easier access
data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Verify the column names
print(data.columns)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean tweets
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.split()
    tweet = [word for word in tweet if word.lower() not in stop_words]
    tweet = ' '.join(tweet)
    return tweet

# Apply cleaning function to the 'text' column
data['cleaned_tweet'] = data['text'].apply(clean_tweet)

# Display the first few cleaned tweets
print(data[['text', 'cleaned_tweet']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer with fewer features
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['cleaned_tweet']).toarray()
y = data['target']

print(X.shape)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, h, y):
        m = len(y)
        return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def gradient_descent(self, X, h, y):
        m = len(y)
        return (1/m) * np.dot(X.T, (h - y))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = self.gradient_descent(X, h, y)
            self.theta -= self.learning_rate * gradient
            cost = self.cost_function(h, y)
            if _ % 100 == 0:
                print(f'Cost at iteration {_}: {cost}')

    def predict(self, X):
        z = np.dot(X, self.theta)
        return self.sigmoid(z)

# Train the model
model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)



# Make predictions
y_pred = model.predict(X)
y_pred_label = [1 if i >= 0.5 else 0 for i in y_pred]

# Evaluate the model
accuracy = accuracy_score(y, y_pred_label)
print(f'Accuracy: {accuracy}')
print(classification_report(y, y_pred_label))
