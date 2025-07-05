import pandas as pd

# Load dataset
df = pd.read_csv('heart_2020_uncleaned.csv')

# View first few rows
df.head()

# Check for missing values
df.info()

# Drop rows with missing values (can be replaced with imputation if needed)
df.dropna(inplace=True)

# Convert categorical 'Yes'/'No' and similar to binary
binary_cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke',
               'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

df.head()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Split features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


import joblib

# Save the model and columns used
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')