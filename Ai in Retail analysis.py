# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load the dataset
# -------------------------------
try:
    df = pd.read_csv('AI_in_Retail_Dataset.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('AI_in_Retail_Dataset.csv', encoding='latin1')

# Display the first few rows
print("Initial Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing Values Per Column:")
print(df.isna().sum())

# -------------------------------
# Handle Missing Values
# -------------------------------
numeric_columns = df.select_dtypes(include=['number'])
df[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.mean())

non_numeric_columns = df.select_dtypes(exclude=['number'])
df[non_numeric_columns.columns] = non_numeric_columns.fillna("Unknown")

# -------------------------------
# Feature Selection and Encoding
# -------------------------------
features = [
    'Age', 'Annual_Salary', 'Gender', 'Education', 'Living_Region',
    'Online_Service_Preference', 'AI_Endorsement', 'AI_Privacy_No_Trust',
    'Payment_Method_Credit/Debit', 'Payment_Method_COD', 'Payment_Method_Ewallet',
    'Product_Category_Appliances', 'Product_Category_Electronics',
    'Product_Category_Groceries', 'Product_Category_Personal_Care',
    'Product_Category_Clothing'
]

# One-hot encode categorical features
X = pd.get_dummies(df[features])

# Encode target
y = df['AI_Satisfication']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30, random_state=42)

# -------------------------------
# Logistic Regression Model
# -------------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
train_accuracy = log_reg.score(X_train, y_train)
test_accuracy = log_reg.score(X_test, y_test)

print(f"\nLogistic Regression - Train Accuracy: {train_accuracy:.4f}")
print(f"Logistic Regression - Test Accuracy: {test_accuracy:.4f}")

mse = mean_squared_error(y_test, y_pred_log)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_log)

print(f"Logistic Regression - MSE: {mse:.4f}")
print(f"Logistic Regression - RMSE: {rmse:.4f}")
print(f"Logistic Regression - R²: {r2:.4f}")

# -------------------------------
# Confusion Matrix
# -------------------------------
conf_matrix = confusion_matrix(y_test, y_pred_log)
sns.heatmap(conf_matrix, annot=True, cmap='viridis', fmt='d')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# -------------------------------
# Classification Report
# -------------------------------
print("\nClassification Report - Logistic Regression:")
print(classification_report(y_test, y_pred_log, target_names=label_encoder.classes_))
