import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("ilpd.csv")

# Add headers if missing
df.columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
              'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Proteins',
              'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']

# Handle missing values
df = df.dropna()

# Encode gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0

# Target variable conversion (1 = liver disease, 2 = no disease)
df['target'] = df['Dataset'].map({1: 1, 2: 0})

# Split data
X = df.drop(['Dataset', 'target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved as model.pkl")
