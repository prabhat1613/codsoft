import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

file_path = 'Churn_Modelling_1.csv'
data = pd.read_csv(file_path)

print("Model training started....")

imputer = SimpleImputer(strategy='mean')
data['Age'] = imputer.fit_transform(data[['Age']])

label_encoder_gender = LabelEncoder()
label_encoder_geography = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
data['Geography'] = label_encoder_geography.fit_transform(data['Geography'])

scaler = StandardScaler()
numerical_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
target = 'Exited'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)

models = {
    'Logistic Regression': f1_lr,
    'Random Forest': f1_rf,
    'Gradient Boosting': f1_gb
}

best_model_name = max(models, key=models.get)

if best_model_name == 'Logistic Regression':
    best_model = lr_model
    feature_importances = None 
elif best_model_name == 'Random Forest':
    best_model = rf_model
    feature_importances = rf_model.feature_importances_
else:
    best_model = gb_model
    feature_importances = gb_model.feature_importances_

joblib.dump(best_model, 'best_model.pkl')
joblib.dump(label_encoder_gender, 'label_encoder_gender.pkl')
joblib.dump(label_encoder_geography, 'label_encoder_geography.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel trained")

best_model = joblib.load('best_model.pkl')
label_encoder_gender = joblib.load('label_encoder_gender.pkl')
label_encoder_geography = joblib.load('label_encoder_geography.pkl')
scaler = joblib.load('scaler.pkl')

new_data = pd.DataFrame({
    'CreditScore': [600, 850],
    'Geography': ['France', 'Spain'],
    'Gender': ['Female', 'Male'],
    'Age': [40, 50],
    'Tenure': [5, 10],
    'Balance': [60000, 120000],
    'NumOfProducts': [2, 1],
    'HasCrCard': [1, 0],
    'IsActiveMember': [1, 0],
    'EstimatedSalary': [50000, 100000]
})

new_data['Gender'] = label_encoder_gender.transform(new_data['Gender'])
new_data['Geography'] = label_encoder_geography.transform(new_data['Geography'])
new_data[numerical_features] = scaler.transform(new_data[numerical_features])

predictions = best_model.predict(new_data)

customer_loss_rate = (predictions.sum() / len(predictions)) * 100
print(f'\nCustomer Loss Rate: {customer_loss_rate:.2f}%')

prediction_labels = ['Churn' if pred == 1 else 'No Churn' for pred in predictions]
new_data['Prediction'] = prediction_labels

if feature_importances is not None:
    feature_importances_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances * 100
    }).sort_values(by='Importance', ascending=False)

    feature_importances_df['Importance'] = feature_importances_df['Importance'].map(lambda x: f'{x:.2f}%')

    print("\nFactors Responsible for the loss of customers based on the data:\n")
    print(feature_importances_df)
