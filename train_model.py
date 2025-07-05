import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('heart_2020_uncleaned.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Convert 'Yes'/'No' to binary
binary_cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke',
               'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encode other categorical features
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a smaller Random Forest model
model = RandomForestClassifier(
    n_estimators=50,    # Reduced number of trees
    max_depth=10,       # Limit tree depth to avoid large model size
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model with compression to reduce file size
joblib.dump(model, 'heart_disease_model.pkl', compress=3)

# Save the list of input feature columns
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

# Optional: Check final model file size
import os
print("Model size (MB):", os.path.getsize("heart_disease_model.pkl") / 1024**2)
