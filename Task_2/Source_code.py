import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

print("Model training started.....\n")
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    rows = [line.strip().split(' ::: ') for line in lines]
    return pd.DataFrame(rows, columns=['ID', 'Title', 'Genre', 'Description'])

file_path = 'train_data.txt'
df = load_data(file_path)

X = df['Description']
y = df['Genre']


df = df.sample(frac=0.1, random_state=42)

X = df['Description']
y = df['Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=100, solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),  
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10)  
}

best_model = None
best_accuracy = 0

for model_name, model in models.items():
    pipeline = make_pipeline(TfidfVectorizer(), model)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

def predict_genre(description, model):
    return model.predict([description])[0]


best_model.fit(X, y)

print("Model training completed!\n")
try:
    while True:
        new_description = input("Enter a movie description (or type 'exit' to quit): ")
        if new_description.lower() == 'exit':
            break
        predicted_genre = predict_genre(new_description, best_model)
        print(f"\nPredicted Genre: {predicted_genre}\n")
except Exception as e:
    print(f"An error occurred: {e}\n")
