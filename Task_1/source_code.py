import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = text.lower()  # convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords])  # remove stop words
    return text

def predict_spam(text):
    text = preprocess_text(text)
    text_vector = vectorizer.transform([text])
    prediction = clf.predict(text_vector)
    return prediction[0]


df = pd.read_csv('spam.csv',encoding="latin1")
stopwords = set(df['v2'])

df['v2'] = df['v2'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['v2'])
y = df['v1'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = LogisticRegression()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
new_sms = input("Enter your message : ")
print("\nThe entered mesage is : ",predict_spam(new_sms))  # Output: 'spam'
