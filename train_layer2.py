# train_layer2.py
# Layer 2: Anemia & Thalassemia Classifier
#
# Since LabQAR is a QA reference dataset (not row-by-row patient data),
# we generate synthetic training data from WHO + Mentzer Index rules.
# This is standard practice in clinical ML when labeled data is scarce.

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

print("🧬 Generating training data from WHO + Mentzer Index rules...")

np.random.seed(42)
N = 2000  # number of synthetic patients

# ── Step 1: Generate realistic CBC ranges ──
ages    = np.random.randint(7, 82, N)
genders = np.random.choice([0, 1], N)          # 0=female, 1=male
hb      = np.random.uniform(6.0, 18.0, N)      # Hemoglobin g/dL
mcv     = np.random.uniform(55.0, 105.0, N)    # Mean Corpuscular Volume fL
rbc     = np.random.uniform(2.5, 6.5, N)       # RBC millions/µL

# ── Step 2: Compute Mentzer Index ──
mentzer = mcv / (rbc + 0.001)

# ── Step 3: Personalized Hb threshold (LabQAR logic) ──
# WHO normal ranges adjusted for age and gender
hb_threshold = np.where(
    genders == 0,                              # female
    np.where(ages > 50, 11.5, 12.0),          # post-menopausal vs normal
    np.where(ages < 18, 12.0, 13.0)           # adolescent vs adult male
)

# ── Step 4: Label each synthetic patient ──
# Rules come directly from WHO and Mentzer Index research
labels = []
for i in range(N):
    anemic = hb[i] < hb_threshold[i]

    if not anemic:
        labels.append("Normal")

    elif mcv[i] < 80 and mentzer[i] < 13:
        # Small RBCs + low Mentzer = Thalassemia trait
        labels.append("Thalassemia")

    elif mcv[i] < 80 and mentzer[i] >= 13:
        # Small RBCs + high Mentzer = Iron Deficiency
        labels.append("Iron_Deficiency")

    elif mcv[i] > 100:
        # Large RBCs = B12 or Folate deficiency
        labels.append("B12_Deficiency")

    else:
        # Low Hb, normal MCV = other anemia
        labels.append("Other_Anemia")

# ── Step 5: Build DataFrame ──
df = pd.DataFrame({
    "age":      ages,
    "gender":   genders,
    "hb":       hb,
    "mcv":      mcv,
    "rbc":      rbc,
    "mentzer":  mentzer,
    "hb_threshold": hb_threshold,
    "hb_deficit":   hb_threshold - hb,   # how far below normal
    "label":    labels
})

print(f"✅ Generated {N} synthetic patients")
print(f"📊 Label distribution:\n{df['label'].value_counts()}")

# ── Step 6: Prepare features and target ──
FEATURES = ["age", "gender", "hb", "mcv", "rbc", "mentzer",
            "hb_threshold", "hb_deficit"]

X = df[FEATURES]
y = df["label"]

# ── Step 7: Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📚 Train: {len(X_train)} | Test: {len(X_test)}")

# ── Step 8: Train Decision Tree ──
# Decision Tree is ideal here — it mirrors the exact
# if/else logic of medical diagnosis rules
print("\n🤖 Training Decision Tree model...")
model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ── Step 9: Evaluate ──
print("\n📈 Model Performance on Test Data:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ── Step 10: Feature importance ──
print("🔍 Top Features the Model Learned:")
importances = pd.Series(model.feature_importances_, index=FEATURES)
print(importances.sort_values(ascending=False).to_string())

# ── Step 11: Save ──
joblib.dump(model,    "models/layer2_model.pkl")
joblib.dump(FEATURES, "models/layer2_features.pkl")

print("\n✅ Layer 2 model saved to models/layer2_model.pkl")
print("🚀 Anemia + Thalassemia classifier ready!")