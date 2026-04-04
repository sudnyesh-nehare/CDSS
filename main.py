# main.py
# Project CDSS — SymbiansLab
# FastAPI server — 3 ML layers, 11 organ conditions
# Founder: Sudnyesh Nehare

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ── Load all models at startup ──
print("🔄 Loading ML models...")

# Layer 1: Infection (Random Forest — trained on CBCovid19EC)
l1_model     = joblib.load("models/layer1_model.pkl")
l1_encoder   = joblib.load("models/layer1_encoder.pkl")
l1_features  = joblib.load("models/layer1_features.pkl")

# Layer 2: Anemia (Decision Tree — trained on WHO + Mentzer rules)
l2_model     = joblib.load("models/layer2_model.pkl")
l2_features  = joblib.load("models/layer2_features.pkl")

# Layer 3: Organ Stress (11 x Logistic Regression — PhysioNet patterns)
l3_models     = joblib.load("models/layer3_models.pkl")
l3_scaler     = joblib.load("models/layer3_scaler.pkl")
l3_features   = joblib.load("models/layer3_features.pkl")
l3_conditions = joblib.load("models/layer3_conditions.pkl")

print("✅ All models loaded. CDSS is ready.")

# ── App ──
app = FastAPI(
    title="Project CDSS — SymbiansLab",
    description="Clinical Decision Support System — ML-powered blood report analyzer. 3 layers, 11 organ conditions.",
    version="3.0.0"
)

# ── Input Schema ──
class CBCReport(BaseModel):
    age: int
    gender: str                            # "male" or "female"
    hb: float                              # Hemoglobin (g/dL)
    wbc: float                             # White Blood Cells (thousands/µL)
    platelets: float                       # Platelets (lakhs)
    neutrophilsP: Optional[float] = None   # Neutrophils %
    lymphocytesP: Optional[float] = None   # Lymphocytes %
    mcv: Optional[float] = None            # Mean Corpuscular Volume (fL)
    rbc: Optional[float] = None            # RBC count (millions/µL)
    monocytesP: Optional[float] = None     # Monocytes %
    eosinophilsP: Optional[float] = None   # Eosinophils %
    basophilsP: Optional[float] = None     # Basophils %
    mpv: Optional[float] = None            # Mean Platelet Volume

# ════════════════════════════════════════════════════════
# FEATURE BUILDERS — one per layer
# ════════════════════════════════════════════════════════

def build_layer1_input(data: dict) -> pd.DataFrame:
    neutro  = data.get("neutrophilsP") or 60.0
    lympho  = data.get("lymphocytesP") or 30.0
    nlr     = neutro / (lympho + 0.001)
    mcv     = data.get("mcv") or 85.0
    rbc     = data.get("rbc") or 5.0
    mentzer = mcv / (rbc + 0.001)
    gender  = 1 if str(data.get("gender", "")).lower() == "male" else 0

    row = {
        "leukocytes":    data.get("wbc", 7.0),
        "neutrophilsP":  neutro,
        "lymphocytesP":  lympho,
        "monocytesP":    data.get("monocytesP") or 7.0,
        "eosinophilsP":  data.get("eosinophilsP") or 2.0,
        "basophilsP":    data.get("basophilsP") or 0.5,
        "hemoglobin":    data.get("hb", 13.0),
        "mcv":           mcv,
        "platelets":     data.get("platelets", 2.5),
        "redbloodcells": rbc,
        "age":           data.get("age", 30),
        "sex_encoded":   gender,
        "nlr":           nlr,
        "mentzer":       mentzer,
    }
    return pd.DataFrame([row])[l1_features]


def build_layer2_input(data: dict) -> pd.DataFrame:
    gender  = 1 if str(data.get("gender", "")).lower() == "male" else 0
    age     = data.get("age", 30)
    hb      = data.get("hb", 13.0)
    mcv     = data.get("mcv") or 85.0
    rbc     = data.get("rbc") or 5.0
    mentzer = mcv / (rbc + 0.001)

    # Personalized Hb threshold (LabQAR WHO logic)
    if gender == 0:
        threshold = 11.5 if age > 50 else 12.0
    else:
        threshold = 12.0 if age < 18 else 13.0

    row = {
        "age":           age,
        "gender":        gender,
        "hb":            hb,
        "mcv":           mcv,
        "rbc":           rbc,
        "mentzer":       mentzer,
        "hb_threshold":  threshold,
        "hb_deficit":    threshold - hb,
    }
    return pd.DataFrame([row])[l2_features]


def build_layer3_input(data: dict, nlr: float) -> pd.DataFrame:
    gender = 1 if str(data.get("gender", "")).lower() == "male" else 0
    row = {
        "age":       data.get("age", 30),
        "gender":    gender,
        "hb":        data.get("hb", 13.0),
        "wbc":       data.get("wbc", 7.0),
        "platelets": data.get("platelets", 2.5),
        "nlr":       nlr,
        "mcv":       data.get("mcv") or 85.0,
        "rbc":       data.get("rbc") or 5.0,
        "lympho":    data.get("lymphocytesP") or 30.0,
        "neutro":    data.get("neutrophilsP") or 60.0,
        "mpv":       data.get("mpv") or 9.0,
    }
    return pd.DataFrame([row])[l3_features]

# ════════════════════════════════════════════════════════
# PRESCRIPTION GENERATOR
# ════════════════════════════════════════════════════════

def generate_prescription(l1: dict, l2: dict, l3: dict) -> dict:
    exercise    = "Moderate walking 30 min/day"
    meal        = "Balanced diet with whole grains and vegetables"
    environment = "Normal home rest"
    warnings    = []

    # ── Critical alerts first ──
    if l3.get("leukemia"):
        warnings.append("⚠️  Leukemia signal detected — consult a hematologist immediately.")
        exercise    = "FULL REST — No physical activity"
        environment = "Urgent: visit a hematologist or oncologist"

    if l3.get("sepsis"):
        warnings.append("⚠️  Sepsis risk — emergency care needed.")
        exercise    = "FULL REST — No physical activity"
        environment = "Go to emergency care immediately"

    if l3.get("cardiac"):
        warnings.append("⚠️  Cardiac strain detected — avoid exertion.")
        exercise    = "Light rest only — no gym or running"
        environment = "Monitor heart rate. Avoid physical stress."

    # ── Infection guidance ──
    if l1["infection_detected"]:
        if l1["risk_level"] == "High":
            exercise    = "FULL REST"
            environment = "Clinical monitoring required"
        else:
            exercise    = "Rest — no gym or running"
            meal        = "Anti-inflammatory: turmeric, ginger, citrus, warm fluids"
            environment = "Cool quiet room. Stay hydrated."

    # ── Anemia guidance ──
    if l2["anemia_type"] == "Iron_Deficiency":
        meal = "Iron-rich: spinach, lentils, jaggery + Vitamin C (lemon juice)"

    elif l2["anemia_type"] == "Thalassemia":
        meal = "Avoid iron supplements. Folate-rich foods: green leafy vegetables."
        warnings.append("ℹ️  Thalassemia trait detected — do NOT self-medicate with iron.")

    elif l2["anemia_type"] == "B12_Deficiency":
        meal = "B12 sources: eggs, dairy, meat. Consider B12 supplement."

    # ── Organ-specific guidance ──
    if l3.get("metabolic"):
        exercise = "Zone 2 Cardio — 40 min brisk walking daily"
        meal     = "Low glycemic diet: reduce sugar, white rice, maida"
        warnings.append("ℹ️  Pre-diabetes / metabolic syndrome pattern detected.")

    if l3.get("liver"):
        meal = "Avoid alcohol. Light diet: fruits, vegetables, avoid fatty foods."
        warnings.append("ℹ️  Liver stress signal — avoid hepatotoxic substances.")

    if l3.get("thyroid"):
        meal = "Iodine-rich foods: fish, dairy, iodized salt. Get TSH test."
        warnings.append("ℹ️  Hypothyroid signal — get thyroid function test (TSH).")

    if l3.get("polycythemia"):
        exercise    = "Light activity only — avoid dehydration"
        environment = "Stay hydrated. Avoid high altitude."
        warnings.append("⚠️  High RBC count — blood clot risk. Consult a doctor.")

    if l3.get("autoimmune"):
        warnings.append("ℹ️  Autoimmune signal — consider ANA / RA factor blood test.")

    if l3.get("respiratory"):
        environment = "Avoid dusty/polluted environments. Deep breathing exercises."
        warnings.append("ℹ️  Respiratory fatigue pattern — check oxygen saturation.")

    if l3.get("vitd"):
        meal = meal + " + Vitamin D: sunlight 20 min/day, eggs, fortified milk."
        warnings.append("ℹ️  Low Vitamin D signal — consider Vit D3 supplement.")

    if l3.get("kidney"):
        warnings.append("ℹ️  Kidney stress signal — stay hydrated. Get creatinine test.")

    return {
        "exercise_plan":    exercise,
        "meal_plan":        meal,
        "environment_plan": environment,
        "warnings":         warnings,
    }

# ════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status":     "CDSS API is live ✅",
        "lab":        "SymbiansLab",
        "founder":    "Sudnyesh Nehare",
        "version":    "3.0.0",
        "layers":     3,
        "conditions": 11,
        "models":     "Layer1(RandomForest) + Layer2(DecisionTree) + Layer3(LogisticRegression x11)"
    }


@app.post("/analyze")
def analyze(report: CBCReport):
    try:
        data = report.model_dump()

        # ── Layer 1: Infection Detection ──
        l1_input = build_layer1_input(data)
        l1_pred  = l1_model.predict(l1_input)[0]
        l1_proba = l1_model.predict_proba(l1_input)[0]
        l1_label = l1_encoder.inverse_transform([l1_pred])[0]
        l1_conf  = round(float(max(l1_proba)) * 100, 1)

        neutro = data.get("neutrophilsP") or 60.0
        lympho = data.get("lymphocytesP") or 30.0
        nlr    = round(neutro / (lympho + 0.001), 2)

        if l1_label == "positive":
            inf_type   = "Viral (COVID-19 / Flu)" if nlr < 6 else "Bacterial / Sepsis Risk"
            risk_level = "High" if nlr > 6 else "Medium"
        else:
            inf_type   = "None"
            risk_level = "Low"

        layer1 = {
            "infection_detected": l1_label == "positive",
            "infection_type":     inf_type,
            "risk_level":         risk_level,
            "nlr":                nlr,
            "model_confidence":   f"{l1_conf}%",
        }

        # ── Layer 2: Anemia Classification ──
        l2_input = build_layer2_input(data)
        l2_pred  = l2_model.predict(l2_input)[0]
        l2_proba = l2_model.predict_proba(l2_input)[0]
        l2_conf  = round(float(max(l2_proba)) * 100, 1)

        layer2 = {
            "anemia_detected":  l2_pred != "Normal",
            "anemia_type":      l2_pred,
            "model_confidence": f"{l2_conf}%",
        }

        # ── Layer 3: 11 Organ Conditions ──
        l3_input  = build_layer3_input(data, nlr)
        l3_scaled = l3_scaler.transform(l3_input)

        layer3         = {}
        active_alerts  = []

        for condition in l3_conditions:
            if condition in l3_models:
                predicted = bool(l3_models[condition].predict(l3_scaled)[0])
                layer3[condition] = predicted
                if predicted:
                    active_alerts.append(condition)

        layer3["active_alerts"] = active_alerts
        layer3["alert_count"]   = len(active_alerts)

        # ── Prescription ──
        prescription = generate_prescription(layer1, layer2, layer3)

        return {
            "patient_profile": {
                "age":    data["age"],
                "gender": data["gender"]
            },
            "layer1_infection":       layer1,
            "layer2_anemia":          layer2,
            "layer3_organ_stress":    layer3,
            "lifestyle_prescription": prescription,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))