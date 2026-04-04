# train.py
# This script reads the real CBCovid19EC dataset,
# trains a Random Forest ML model on it,
# and saves the trained model to a file.
# You only run this ONCE. The saved model is what the API uses.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("📂 Loading dataset...")
df = pd.read_csv("CBCovid19EC.csv")

# ── Step 1: Clean the data ──
# Remove spaces from text columns
df["sex"] = df["sex"].str.strip().str.lower()
df["pcr"] = df["pcr"].str.strip().str.lower()

print(f"✅ Loaded {len(df)} records")
print(f"📊 PCR distribution:\n{df['pcr'].value_counts()}")

# ── Step 2: Pick the features ML will learn from ──
# These are the CBC values available in a standard blood report
FEATURES = [
    "leukocytes",      # Total WBC
    "neutrophilsP",    # Neutrophils %
    "lymphocytesP",    # Lymphocytes %
    "monocytesP",      # Monocytes %
    "eosinophilsP",    # Eosinophils %
    "basophilsP",      # Basophils %
    "hemoglobin",      # Hb
    "mcv",             # Mean Corpuscular Volume
    "platelets",       # Platelet count
    "redbloodcells",   # RBC count
    "age",             # Patient age
]

# ── Step 3: Encode gender as a number (ML needs numbers not text) ──
df["sex_encoded"] = (df["sex"] == "male").astype(int)  # male=1, female=0
FEATURES.append("sex_encoded")

# ── Step 4: Add NLR as an engineered feature ──
# NLR = Neutrophil-to-Lymphocyte Ratio
# This is the key "Disease Signature" from CBCovid19EC research
df["nlr"] = df["neutrophilsP"] / (df["lymphocytesP"] + 0.001)
FEATURES.append("nlr")

# ── Step 5: Add Mentzer Index as an engineered feature ──
# Mentzer = MCV / RBC — detects Thalassemia vs Iron Deficiency
df["mentzer"] = df["mcv"] / (df["redbloodcells"] + 0.001)
FEATURES.append("mentzer")

# ── Step 6: Prepare X (inputs) and y (target) ──
X = df[FEATURES].copy()
y = df["pcr"]  # positive or negative

# Drop rows where any value is missing
X = X.dropna()
y = y[X.index]

print(f"\n🔢 Training on {len(X)} clean records with {len(FEATURES)} features")

# ── Step 7: Encode target (positive=1, negative=0) ──
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"🏷️  Classes: {le.classes_}")  # shows [negative, positive]

# ── Step 8: Split into training and testing ──
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n📚 Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── Step 9: Train the Random Forest model ──
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees voting together
    max_depth=8,        # prevent overfitting on small dataset
    random_state=42,
    class_weight="balanced"  # handles unequal positive/negative counts
)
model.fit(X_train, y_train)

# ── Step 10: Evaluate the model ──
print("\n📈 Model Performance on Test Data:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Accuracy: ",accuracy_score(y_test,y_pred))

# ── Step 11: Show which features matter most ──
print("🔍 Top Features the Model Learned:")
importances = pd.Series(model.feature_importances_, index=FEATURES)
print(importances.sort_values(ascending=False).head(8).to_string())

# ── Step 12: Save the trained model and label encoder ──
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/layer1_model.pkl")
joblib.dump(le, "models/layer1_encoder.pkl")
joblib.dump(FEATURES, "models/layer1_features.pkl")

print("\n✅ Model saved to models/layer1_model.pkl")
print("🚀 Ready to use in the API!")