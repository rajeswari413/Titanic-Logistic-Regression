 # Project 4: Titanic Survival Prediction using Logistic Regression
# Objective: Predict passenger survival based on Pclass, Sex, Age, and Fare.
# Model: Logistic Regression
# Accuracy: Approximately 80.45%
import pandas as pd

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())
# Select useful columns
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing Age values with the median
df['Age'] = df['Age'].fillna(df['Age'].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Convert Sex into numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

print("\nDataset after converting Sex:")
print(df.head())
# Separate features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")
# Make predictions on test data
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])
from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))