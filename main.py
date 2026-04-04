# main.py
# Project CDSS - SymbiansLab
# FastAPI server - 3 ML layers, 11 organ conditions, English + Hindi output
# Founder: Sudnyesh Nehare

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# -- Load all models at startup --
print("Loading ML models...")

# Layer 1: Infection (Random Forest - trained on CBCovid19EC)
l1_model      = joblib.load("models/layer1_model.pkl")
l1_encoder    = joblib.load("models/layer1_encoder.pkl")
l1_features   = joblib.load("models/layer1_features.pkl")

# Layer 2: Anemia (Decision Tree - trained on WHO + Mentzer rules)
l2_model      = joblib.load("models/layer2_model.pkl")
l2_features   = joblib.load("models/layer2_features.pkl")

# Layer 3: Organ Stress (11 x Logistic Regression - PhysioNet patterns)
l3_models     = joblib.load("models/layer3_models.pkl")
l3_scaler     = joblib.load("models/layer3_scaler.pkl")
l3_features   = joblib.load("models/layer3_features.pkl")
l3_conditions = joblib.load("models/layer3_conditions.pkl")

print("All models loaded. CDSS is ready.")

# -- App --
app = FastAPI(
    title="Project CDSS - SymbiansLab",
    description="Clinical Decision Support System - ML-powered blood report analyzer. 3 layers, 11 organ conditions, English + Hindi output.",
    version="4.0.0"
)

# -- Input Schema --
class CBCReport(BaseModel):
    age: int
    gender: str                            # "male" or "female"
    hb: float                              # Hemoglobin (g/dL)
    wbc: float                             # White Blood Cells (thousands/uL)
    platelets: float                       # Platelets (lakhs)
    neutrophilsP: Optional[float] = None   # Neutrophils %
    lymphocytesP: Optional[float] = None   # Lymphocytes %
    mcv: Optional[float] = None            # Mean Corpuscular Volume (fL)
    rbc: Optional[float] = None            # RBC count (millions/uL)
    monocytesP: Optional[float] = None     # Monocytes %
    eosinophilsP: Optional[float] = None   # Eosinophils %
    basophilsP: Optional[float] = None     # Basophils %
    mpv: Optional[float] = None            # Mean Platelet Volume


# ================================================================
# HINDI TRANSLATIONS
# Simple patient-friendly language - not medical jargon
# ================================================================

HINDI = {
    # Exercise
    "Moderate walking 30 min/day":
        "roz 30 minute aram se tahlen - subah ya shaam ko",
    "FULL REST - No physical activity":
        "pura aram karen - koi bhi kasrat ya mehnat ka kaam na karen",
    "Light rest only - no gym or running":
        "halka aram karen - jim ya daudna bilkul band rakhen",
    "Rest - no gym or running":
        "aram karen - jim ya daudna na karen",
    "Zone 2 Cardio - 40 min brisk walking daily":
        "roz 40 minute tez chaal se chalen - sans thodi foole par baat kar saken",
    "Light activity only - avoid dehydration":
        "halki gatividhi karen - paani khoob pien, sharir ko sookhne na den",

    # Meal
    "Balanced diet with whole grains and vegetables":
        "saabut anaaj aur sabziyon se bhara santulit khaana khaen",
    "Iron-rich: spinach, lentils, jaggery + Vitamin C (lemon juice)":
        "palak, daal, gud khaen aur saath mein nimbu ka ras len - isse khoon banta hai",
    "Avoid iron supplements. Folate-rich foods: green leafy vegetables.":
        "aayron ki goliyaan na len - hari pattedaar sabziyan jaise palak aur methi khaen",
    "B12 sources: eggs, dairy, meat. Consider B12 supplement.":
        "ande, doodh, dahi khaen - doctor se B12 ki goli lene ki salah len",
    "Anti-inflammatory: turmeric, ginger, citrus, warm fluids":
        "haldi wala doodh, adrak ki chai, nimbu paani aur garm taral padarth len",
    "Low glycemic diet: reduce sugar, white rice, maida":
        "cheeni, safed chawal aur maida kam karen - brown rice aur daalen khaen",
    "Avoid alcohol. Light diet: fruits, vegetables, avoid fatty foods.":
        "sharaab bilkul band karen - phal aur sabziyan khaen, tala hua khaana na khaen",
    "Iodine-rich foods: fish, dairy, iodized salt. Get TSH test.":
        "aayodeen yukt namak, machli aur doodh len - thyroid ki jaanch (TSH) karwaen",

    # Meal with Vitamin D suffix
    "Avoid iron supplements. Folate-rich foods: green leafy vegetables. + Vitamin D: sunlight 20 min/day, eggs, fortified milk.":
        "aayron ki goliyaan na len - palak, methi khaen. Roz 20 minute dhoop len aur D3 ki goli len",
    "Iron-rich: spinach, lentils, jaggery + Vitamin C (lemon juice) + Vitamin D: sunlight 20 min/day, eggs, fortified milk.":
        "palak, daal, gud aur nimbu khaen. Roz 20 minute dhoop len aur D3 ki goli len",
    "B12 sources: eggs, dairy, meat. Consider B12 supplement. + Vitamin D: sunlight 20 min/day, eggs, fortified milk.":
        "ande, doodh, dahi khaen - B12 aur D3 dono ki goli doctor se len",
    "Low glycemic diet: reduce sugar, white rice, maida + Vitamin D: sunlight 20 min/day, eggs, fortified milk.":
        "cheeni aur maida kam karen - brown rice khaen. Roz 20 minute dhoop len",
    "Balanced diet with whole grains and vegetables + Vitamin D: sunlight 20 min/day, eggs, fortified milk.":
        "saabut anaaj aur sabziyan khaen. Roz 20 minute dhoop len aur D3 ki goli len",

    # Environment
    "Normal home rest":
        "ghar par aram karen - koi chinta nahin",
    "Go to emergency care immediately":
        "turant aspatal ke emergency vibhag mein jaen",
    "Urgent: visit a hematologist or oncologist":
        "jald se jald khoon ke doctor (hematologist) se milen",
    "Monitor heart rate. Avoid physical stress.":
        "dil ki dhadkan par dhyaan den - sharirik aur maansik tanaav se bachen",
    "Clinical monitoring required":
        "doctor ki nigrani mein rahen - akele ghar par na rahen",
    "Cool quiet room. Stay hydrated.":
        "thande aur shaant kamre mein rahen - har ghante paani pien",
    "Avoid dusty/polluted environments. Deep breathing exercises.":
        "dhool aur pradushan se door rahen - roz gehri saans ki kasrat karen",
    "Stay hydrated. Avoid high altitude.":
        "khoob paani pien - pahadi jagahon par jaane se bachen",

    # Warnings
    "WARNING: Leukemia signal detected - consult a hematologist immediately.":
        "CHETAN: khoon ke cancer ka sanket - turant khoon ke doctor se milen",
    "WARNING: Sepsis risk - emergency care needed.":
        "CHETAN: khoon mein gambhir sankraman ka khatra - abhi emergency jaen",
    "WARNING: Cardiac strain detected - avoid exertion.":
        "CHETAN: dil par dabaav hai - koi bhi bhaari kaam na karen",
    "INFO: Thalassemia trait detected - do NOT self-medicate with iron.":
        "JAANKARI: thalassemia ka sanket - bina doctor ke aayron ki dawaa bilkul na len",
    "INFO: Pre-diabetes / metabolic syndrome pattern detected.":
        "JAANKARI: sugar badhne ka khatra hai - abhi se khaan-paan sudhaaren",
    "INFO: Liver stress signal - avoid hepatotoxic substances.":
        "JAANKARI: liver par dabaav hai - sharaab aur bhaari dawaen band karen",
    "INFO: Hypothyroid signal - get thyroid function test (TSH).":
        "JAANKARI: thyroid kam kaam kar raha hai - TSH jaanch karwaen",
    "WARNING: High RBC count - blood clot risk. Consult a doctor.":
        "CHETAN: khoon gaadha hai - khoon ke thaake ka khatra, doctor se milen",
    "INFO: Autoimmune signal - consider ANA / RA factor blood test.":
        "JAANKARI: sharir khud par hamla kar sakta hai - ANA aur RA factor jaanch karwaen",
    "INFO: Respiratory fatigue pattern - check oxygen saturation.":
        "JAANKARI: pheprhde thake hue hain - oxygen ka sthar (SpO2) jaanchen",
    "INFO: Low Vitamin D signal - consider Vit D3 supplement.":
        "JAANKARI: vitamin D kam hai - roz 20 minute dhoop len aur D3 ki goli len",
    "INFO: Kidney stress signal - stay hydrated. Get creatinine test.":
        "JAANKARI: kidney par dabaav hai - khoob paani pien aur creatinine jaanch karwaen",
}


def translate(text: str) -> str:
    return HINDI.get(text, "kripya doctor se salah len")


# ================================================================
# FEATURE BUILDERS
# ================================================================

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

    if gender == 0:
        threshold = 11.5 if age > 50 else 12.0
    else:
        threshold = 12.0 if age < 18 else 13.0

    row = {
        "age":          age,
        "gender":       gender,
        "hb":           hb,
        "mcv":          mcv,
        "rbc":          rbc,
        "mentzer":      mentzer,
        "hb_threshold": threshold,
        "hb_deficit":   threshold - hb,
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


# ================================================================
# PRESCRIPTION GENERATOR
# ================================================================

def generate_prescription(l1: dict, l2: dict, l3: dict) -> dict:
    exercise    = "Moderate walking 30 min/day"
    meal        = "Balanced diet with whole grains and vegetables"
    environment = "Normal home rest"
    warnings    = []

    # Critical alerts first
    if l3.get("leukemia"):
        warnings.append("WARNING: Leukemia signal detected - consult a hematologist immediately.")
        exercise    = "FULL REST - No physical activity"
        environment = "Urgent: visit a hematologist or oncologist"

    if l3.get("sepsis"):
        warnings.append("WARNING: Sepsis risk - emergency care needed.")
        exercise    = "FULL REST - No physical activity"
        environment = "Go to emergency care immediately"

    if l3.get("cardiac"):
        warnings.append("WARNING: Cardiac strain detected - avoid exertion.")
        exercise    = "Light rest only - no gym or running"
        environment = "Monitor heart rate. Avoid physical stress."

    # Infection
    if l1["infection_detected"]:
        if l1["risk_level"] == "High":
            exercise    = "FULL REST - No physical activity"
            environment = "Clinical monitoring required"
        else:
            exercise    = "Rest - no gym or running"
            meal        = "Anti-inflammatory: turmeric, ginger, citrus, warm fluids"
            environment = "Cool quiet room. Stay hydrated."

    # Anemia
    if l2["anemia_type"] == "Iron_Deficiency":
        meal = "Iron-rich: spinach, lentils, jaggery + Vitamin C (lemon juice)"
    elif l2["anemia_type"] == "Thalassemia":
        meal = "Avoid iron supplements. Folate-rich foods: green leafy vegetables."
        warnings.append("INFO: Thalassemia trait detected - do NOT self-medicate with iron.")
    elif l2["anemia_type"] == "B12_Deficiency":
        meal = "B12 sources: eggs, dairy, meat. Consider B12 supplement."

    # Organ specific
    if l3.get("metabolic"):
        exercise = "Zone 2 Cardio - 40 min brisk walking daily"
        meal     = "Low glycemic diet: reduce sugar, white rice, maida"
        warnings.append("INFO: Pre-diabetes / metabolic syndrome pattern detected.")

    if l3.get("liver"):
        meal = "Avoid alcohol. Light diet: fruits, vegetables, avoid fatty foods."
        warnings.append("INFO: Liver stress signal - avoid hepatotoxic substances.")

    if l3.get("thyroid"):
        meal = "Iodine-rich foods: fish, dairy, iodized salt. Get TSH test."
        warnings.append("INFO: Hypothyroid signal - get thyroid function test (TSH).")

    if l3.get("polycythemia"):
        exercise    = "Light activity only - avoid dehydration"
        environment = "Stay hydrated. Avoid high altitude."
        warnings.append("WARNING: High RBC count - blood clot risk. Consult a doctor.")

    if l3.get("autoimmune"):
        warnings.append("INFO: Autoimmune signal - consider ANA / RA factor blood test.")

    if l3.get("respiratory"):
        environment = "Avoid dusty/polluted environments. Deep breathing exercises."
        warnings.append("INFO: Respiratory fatigue pattern - check oxygen saturation.")

    if l3.get("vitd"):
        meal = meal + " + Vitamin D: sunlight 20 min/day, eggs, fortified milk."
        warnings.append("INFO: Low Vitamin D signal - consider Vit D3 supplement.")

    if l3.get("kidney"):
        warnings.append("INFO: Kidney stress signal - stay hydrated. Get creatinine test.")

    return {
        "exercise_plan": {
            "english": exercise,
            "hindi":   translate(exercise)
        },
        "meal_plan": {
            "english": meal,
            "hindi":   translate(meal)
        },
        "environment_plan": {
            "english": environment,
            "hindi":   translate(environment)
        },
        "warnings": [
            {"english": w, "hindi": translate(w)}
            for w in warnings
        ],
    }


# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/")
def root():
    return {
        "status":     "CDSS API is live",
        "lab":        "SymbiansLab",
        "founder":    "Sudnyesh Nehare",
        "version":    "4.0.0",
        "layers":     3,
        "conditions": 11,
        "languages":  ["english", "hindi"],
        "models":     "Layer1(RandomForest) + Layer2(DecisionTree) + Layer3(LogisticRegression x11)"
    }


@app.post("/analyze")
def analyze(report: CBCReport):
    try:
        data = report.model_dump()

        # Layer 1: Infection Detection
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

        # Layer 2: Anemia Classification
        l2_input = build_layer2_input(data)
        l2_pred  = l2_model.predict(l2_input)[0]
        l2_proba = l2_model.predict_proba(l2_input)[0]
        l2_conf  = round(float(max(l2_proba)) * 100, 1)

        layer2 = {
            "anemia_detected":  l2_pred != "Normal",
            "anemia_type":      l2_pred,
            "model_confidence": f"{l2_conf}%",
        }

        # Layer 3: 11 Organ Conditions
        l3_input  = build_layer3_input(data, nlr)
        l3_scaled = l3_scaler.transform(l3_input)

        layer3        = {}
        active_alerts = []

        for condition in l3_conditions:
            if condition in l3_models:
                predicted = bool(l3_models[condition].predict(l3_scaled)[0])
                layer3[condition] = predicted
                if predicted:
                    active_alerts.append(condition)

        layer3["active_alerts"] = active_alerts
        layer3["alert_count"]   = len(active_alerts)

        # Prescription
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