"""
VrikkaPulse — Centralized Medical Knowledge Layer.

All medical rules, guideline versioning, kidney equations, CGA classification,
chronicity logic, longitudinal-change logic, symptom-safety and the AI safety
validator live here. Nothing medical is hard-coded inside UI components or routes.

Scope: monitoring + education only. NEVER diagnosis / prescription / prediction.
"""
from __future__ import annotations
import math
import re
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# 3/4. Authoritative guideline sources + versioning
# ---------------------------------------------------------------------------
GUIDELINE_SOURCES = {
    "kdigo_2024_ckd": {
        "source": "KDIGO",
        "title": "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of CKD",
        "guideline_version": "2024",
        "url": "https://kdigo.org/guidelines/ckd-evaluation-and-management/kdigo-2024-ckd-guideline/",
        "last_reviewed": "2024-03-01",
        "status": "current",
        "note": "A focused update to Chapter 3 was initiated in 2026; architecture supports versioning.",
    },
    "nkf_ckd_criteria": {
        "source": "National Kidney Foundation (NKF)",
        "title": "NKF — Criteria & Classification of CKD",
        "guideline_version": "current",
        "url": "https://www.kidney.org/what-criteria-ckd",
        "last_reviewed": "2024-01-01",
        "status": "current",
    },
    "ckd_epi_2021": {
        "source": "NKF / ASN Task Force (CKD-EPI 2021, race-free)",
        "title": "CKD-EPI 2021 creatinine & creatinine-cystatin C equations",
        "guideline_version": "2021",
        "url": "https://www.kidney.org/professionals/gfr_calculator",
        "last_reviewed": "2021-09-01",
        "status": "current",
    },
    "who_ai_health": {
        "source": "World Health Organization (WHO)",
        "title": "Ethics & Governance of AI for Health / LMM Guidance",
        "guideline_version": "2024",
        "url": "https://www.who.int/publications/i/item/9789240084759",
        "last_reviewed": "2024-01-01",
        "status": "current",
    },
}

# 4. MedicalRule registry (versioned, non-hardcoded content)
MEDICAL_RULES = [
    {
        "id": "gfr_categories",
        "rule_name": "GFR categories (G1–G5)",
        "source": "KDIGO",
        "guideline_version": "2024",
        "effective_date": "2024-03-01",
        "last_reviewed": "2024-03-01",
        "status": "current",
        "content": "G1 ≥90, G2 60–89, G3a 45–59, G3b 30–44, G4 15–29, G5 <15 mL/min/1.73m².",
    },
    {
        "id": "acr_categories",
        "rule_name": "Albuminuria categories (A1–A3)",
        "source": "KDIGO",
        "guideline_version": "2024",
        "effective_date": "2024-03-01",
        "last_reviewed": "2024-03-01",
        "status": "current",
        "content": "A1 <30 mg/g, A2 30–300 mg/g, A3 >300 mg/g (or mg/mmol equivalents).",
    },
    {
        "id": "ckd_definition",
        "rule_name": "CKD definition (chronicity ≥3 months)",
        "source": "KDIGO / NKF",
        "guideline_version": "2024",
        "effective_date": "2024-03-01",
        "last_reviewed": "2024-03-01",
        "status": "current",
        "content": "CKD = abnormalities of kidney structure/function present ≥3 months with implications for health.",
    },
    {
        "id": "egfr_equations",
        "rule_name": "eGFR estimation (CKD-EPI 2021)",
        "source": "CKD-EPI 2021",
        "guideline_version": "2021",
        "effective_date": "2021-09-01",
        "last_reviewed": "2021-09-01",
        "status": "current",
        "content": "Use CKD-EPI 2021 creatinine, cystatin C, or creatinine-cystatin C equations. eGFR is an estimate, not a direct GFR measurement.",
    },
    {
        "id": "monitoring_frequency",
        "rule_name": "Monitoring frequency (educational)",
        "source": "KDIGO",
        "guideline_version": "2024",
        "effective_date": "2024-03-01",
        "last_reviewed": "2024-03-01",
        "status": "current",
        "content": "For established CKD, GFR & albuminuria are generally assessed at least annually, more often at higher progression risk. Follow the schedule set by your healthcare professional.",
    },
]

DISCLAIMER = ("VrikkaPulse provides monitoring and educational support. It does not "
              "diagnose disease or replace your healthcare professional.")
AI_DISCLAIMER = ("Vrikka AI provides general educational information and does not replace "
                 "professional medical advice.")

# ---------------------------------------------------------------------------
# 6. CGA classification — G category
# ---------------------------------------------------------------------------
def gfr_category(egfr: float | None):
    if egfr is None:
        return None
    e = float(egfr)
    if e >= 90:
        return {"code": "G1", "range": "≥90", "label": "Normal or high"}
    if e >= 60:
        return {"code": "G2", "range": "60–89", "label": "Mildly decreased"}
    if e >= 45:
        return {"code": "G3a", "range": "45–59", "label": "Mildly to moderately decreased"}
    if e >= 30:
        return {"code": "G3b", "range": "30–44", "label": "Moderately to severely decreased"}
    if e >= 15:
        return {"code": "G4", "range": "15–29", "label": "Severely decreased"}
    return {"code": "G5", "range": "<15", "label": "Kidney failure"}


# 6. A category. Accepts value + unit. Converts mg/mmol -> mg/g (×8.84).
def acr_category(value: float | None, unit: str = "mg/g"):
    if value is None:
        return None
    v = float(value)
    u = (unit or "mg/g").lower().replace(" ", "")
    if "mmol" in u:  # mg/mmol -> mg/g
        v = v * 8.84
    if v < 30:
        return {"code": "A1", "range": "<30 mg/g", "label": "Normal to mildly increased"}
    if v <= 300:
        return {"code": "A2", "range": "30–300 mg/g", "label": "Moderately increased"}
    return {"code": "A3", "range": ">300 mg/g", "label": "Severely increased"}


# 7. KDIGO GFR × ACR educational heatmap risk cell (monitoring context only,
#    NOT a personalized failure-probability).
_HEATMAP = {
    ("G1", "A1"): "low", ("G1", "A2"): "moderate", ("G1", "A3"): "high",
    ("G2", "A1"): "low", ("G2", "A2"): "moderate", ("G2", "A3"): "high",
    ("G3a", "A1"): "moderate", ("G3a", "A2"): "high", ("G3a", "A3"): "very_high",
    ("G3b", "A1"): "high", ("G3b", "A2"): "very_high", ("G3b", "A3"): "very_high",
    ("G4", "A1"): "very_high", ("G4", "A2"): "very_high", ("G4", "A3"): "very_high",
    ("G5", "A1"): "very_high", ("G5", "A2"): "very_high", ("G5", "A3"): "very_high",
}
_RISK_LABEL = {
    "low": "Low risk category — monitor per your clinician",
    "moderate": "Moderately increased risk category",
    "high": "High risk category",
    "very_high": "Very high risk category",
}

def heatmap_cell(g_code: str | None, a_code: str | None):
    if not g_code or not a_code:
        return None
    key = (g_code, a_code)
    risk = _HEATMAP.get(key)
    if not risk:
        return None
    return {"g": g_code, "a": a_code, "risk": risk, "label": _RISK_LABEL[risk],
            "context": "Educational KDIGO CGA classification — not a disease prediction."}


def heatmap_matrix():
    gs = ["G1", "G2", "G3a", "G3b", "G4", "G5"]
    as_ = ["A1", "A2", "A3"]
    return {"g_categories": gs, "a_categories": as_,
            "cells": [{"g": g, "a": a, "risk": _HEATMAP[(g, a)]} for g in gs for a in as_]}


# ---------------------------------------------------------------------------
# 8/9. eGFR equations — CKD-EPI 2021 (race-free). sex: "male"/"female".
# ---------------------------------------------------------------------------
def egfr_cr_2021(scr_mg_dl: float, age: float, sex: str):
    if not scr_mg_dl or not age or not sex:
        return None
    female = sex.lower().startswith("f")
    k = 0.7 if female else 0.9
    a = -0.241 if female else -0.302
    r = scr_mg_dl / k
    egfr = 142 * (min(r, 1) ** a) * (max(r, 1) ** -1.200) * (0.9938 ** age)
    if female:
        egfr *= 1.012
    return round(egfr, 1)


def egfr_cys_2021(scys_mg_l: float, age: float, sex: str):
    if not scys_mg_l or not age or not sex:
        return None
    female = sex.lower().startswith("f")
    r = scys_mg_l / 0.8
    egfr = 133 * (min(r, 1) ** -0.499) * (max(r, 1) ** -1.328) * (0.996 ** age)
    if female:
        egfr *= 0.932
    return round(egfr, 1)


def egfr_cr_cys_2021(scr_mg_dl: float, scys_mg_l: float, age: float, sex: str):
    if not scr_mg_dl or not scys_mg_l or not age or not sex:
        return None
    female = sex.lower().startswith("f")
    k = 0.7 if female else 0.9
    a = -0.219 if female else -0.144
    rcr = scr_mg_dl / k
    rcys = scys_mg_l / 0.8
    egfr = (135 * (min(rcr, 1) ** a) * (max(rcr, 1) ** -0.544)
            * (min(rcys, 1) ** -0.323) * (max(rcys, 1) ** -0.778) * (0.9961 ** age))
    if female:
        egfr *= 0.963
    return round(egfr, 1)


def compute_egfr(scr=None, scys=None, age=None, sex=None):
    """Returns best available eGFR estimate with method + safe messaging."""
    if not age or not sex:
        return {"value": None, "method": None,
                "message": "There isn't enough information to calculate this value."}
    if scr and scys:
        return {"value": egfr_cr_cys_2021(scr, scys, age, sex), "method": "eGFRcr-cys (CKD-EPI 2021)",
                "note": "eGFR is an estimate, not a direct measurement of GFR."}
    if scr:
        return {"value": egfr_cr_2021(scr, age, sex), "method": "eGFRcr (CKD-EPI 2021)",
                "note": "eGFR is an estimate, not a direct measurement of GFR."}
    if scys:
        return {"value": egfr_cys_2021(scys, age, sex), "method": "eGFRcys (CKD-EPI 2021)",
                "note": "eGFR is an estimate, not a direct measurement of GFR."}
    return {"value": None, "method": None,
            "message": "There isn't enough information to calculate this value."}


# ---------------------------------------------------------------------------
# 10. Chronicity engine
# ---------------------------------------------------------------------------
CHRONICITY_STATES = {
    "insufficient": "Insufficient longitudinal data",
    "possible": "Possible abnormal finding — needs confirmation",
    "persistent": "Persistent finding documented",
    "clinician_ckd": "Clinician-confirmed CKD",
    "user_ckd": "User-reported CKD",
}

def _days_between(d1: str, d2: str):
    try:
        a = datetime.fromisoformat(d1.replace("Z", "+00:00"))
        b = datetime.fromisoformat(d2.replace("Z", "+00:00"))
        return abs((b - a).days)
    except Exception:
        return None


def chronicity_state(egfr_results, user_reported_ckd=False, clinician_confirmed=False):
    """egfr_results: list of {value, date} sorted or unsorted. Conservative."""
    if clinician_confirmed:
        return {"state": "clinician_ckd", "label": CHRONICITY_STATES["clinician_ckd"],
                "message": "Clinician-confirmed CKD is on record."}
    if user_reported_ckd:
        return {"state": "user_ckd", "label": CHRONICITY_STATES["user_ckd"],
                "message": "User-reported CKD. Discuss confirmation with your healthcare professional."}
    abnormal = [r for r in (egfr_results or []) if r.get("value") is not None and r["value"] < 60]
    if not egfr_results or len(egfr_results) < 2:
        if abnormal:
            return {"state": "possible", "label": CHRONICITY_STATES["possible"],
                    "message": "This is a recorded result. A single measurement cannot establish chronic kidney disease."}
        return {"state": "insufficient", "label": CHRONICITY_STATES["insufficient"],
                "message": "Not enough recorded data over time to assess chronicity yet."}
    if len(abnormal) >= 2:
        dates = sorted([r["date"] for r in abnormal if r.get("date")])
        if len(dates) >= 2:
            span = _days_between(dates[0], dates[-1])
            if span is not None and span >= 90:
                return {"state": "persistent", "label": CHRONICITY_STATES["persistent"],
                        "message": "A persistent finding (≥3 months) is documented. This is a monitoring flag, not a diagnosis. Discuss with your healthcare professional."}
        return {"state": "possible", "label": CHRONICITY_STATES["possible"],
                "message": "Repeat abnormal results recorded, but ≥3-month chronicity is not yet established."}
    return {"state": "possible", "label": CHRONICITY_STATES["possible"],
            "message": "This is a recorded result. A single measurement cannot establish chronic kidney disease."}


# ---------------------------------------------------------------------------
# 11/12. Longitudinal change + AKI/acute-change protection (monitoring flags)
# ---------------------------------------------------------------------------
def longitudinal_flags(prev, latest):
    """prev/latest: {egfr, acr, creatinine, date}. Returns monitoring flags (not diagnoses)."""
    flags = []
    if prev.get("egfr") and latest.get("egfr"):
        change = (latest["egfr"] - prev["egfr"]) / prev["egfr"] * 100
        days = _days_between(prev.get("date", ""), latest.get("date", "")) if prev.get("date") and latest.get("date") else None
        if abs(change) >= 20:
            if days is not None and days < 14:
                flags.append({"type": "acute_change", "severity": "review",
                              "message": "This recorded change may have multiple causes, and chronicity cannot be established from this information alone. Please discuss significant changes with your healthcare professional."})
            else:
                flags.append({"type": "egfr_change", "severity": "review",
                              "message": f"Recorded eGFR changed by {round(change,1)}%. This change is larger than expected measurement variability and may warrant clinical review."})
    if prev.get("acr") and latest.get("acr"):
        if prev["acr"] > 0 and latest["acr"] >= prev["acr"] * 2:
            flags.append({"type": "acr_doubling", "severity": "review",
                          "message": "Recorded ACR doubled compared with a previous result. This may warrant clinical review."})
    return flags


# ---------------------------------------------------------------------------
# 20. Symptom safety layer
# ---------------------------------------------------------------------------
EMERGENCY_SYMPTOMS = {
    "shortness_of_breath": "Shortness of breath",
    "reduced_urine_output": "Reduced urine output",
    "blood_in_urine": "Blood in urine",
    "chest_pain": "Chest pain",
}
CONCERNING_SYMPTOMS = {
    "foot_swelling", "eye_swelling", "fatigue", "foamy_urine", "nausea",
    "loss_of_appetite", "muscle_cramps", "itching",
}

def symptom_safety(selected: list[str]):
    if not selected or "no_symptoms" in selected:
        return {"level": "none", "message": None}
    emergency = [s for s in selected if s in EMERGENCY_SYMPTOMS]
    if emergency:
        return {"level": "urgent",
                "message": "Some symptoms can have multiple causes. If symptoms are severe, sudden, or worsening — or if you feel this may be an emergency — seek urgent medical care or contact your local emergency services."}
    concerning = [s for s in selected if s in CONCERNING_SYMPTOMS]
    if concerning:
        return {"level": "caution",
                "message": "Some symptoms can have multiple causes. If symptoms are severe, sudden, or worsening, seek appropriate medical attention. Please discuss with your healthcare professional."}
    return {"level": "none", "message": None}


# ---------------------------------------------------------------------------
# 34. Wellness & Monitoring Score — NON-CLINICAL. Never uses lab values/symptoms.
# ---------------------------------------------------------------------------
def wellness_score(inputs: dict):
    """
    inputs (all 0..1 consistency ratios over the period):
      checkin_consistency, medication_logging, bp_logging,
      water_tracking, activity_logging, salt_tracking, reminder_adherence
    """
    weights = {
        "checkin_consistency": 0.25,
        "medication_logging": 0.20,
        "bp_logging": 0.15,
        "water_tracking": 0.15,
        "activity_logging": 0.10,
        "salt_tracking": 0.075,
        "reminder_adherence": 0.075,
    }
    total = 0.0
    for k, w in weights.items():
        total += max(0.0, min(1.0, float(inputs.get(k, 0) or 0))) * w
    score = round(total * 100)
    return {
        "score": score,
        "disclaimer": ("This Wellness & Monitoring Score reflects monitoring consistency and "
                       "healthy habits only. It is NOT a medical diagnosis or validated clinical score."),
        "inputs": {k: round(max(0.0, min(1.0, float(inputs.get(k, 0) or 0))), 2) for k in weights},
    }


# ---------------------------------------------------------------------------
# 32. AI safety validator — blocks/rephrases unsafe medical output.
# ---------------------------------------------------------------------------
_UNSAFE_PATTERNS = [
    (r"\byou (?:have|are diagnosed with|are suffering from)\b.{0,40}\b(ckd|kidney (?:failure|disease)|aki)\b", "diagnosis"),
    (r"\byour kidneys? (?:are|is) (?:failing|healthy|fine|damaged)\b", "diagnosis"),
    (r"\byou (?:should|must|need to) (?:start|stop|take|increase|decrease|change|reduce|double)\b.{0,30}\b(medication|dose|drug|pill|tablet|medicine)\b", "medication_change"),
    (r"\b(?:i (?:prescribe|recommend (?:starting|stopping)))\b", "prescription"),
    (r"\byou (?:will|are going to) (?:develop|get) (?:kidney failure|ckd|dialysis)\b", "prediction"),
    (r"\byour kidneys? are (?:completely )?(?:safe|healthy|normal|fine)\b", "false_reassurance"),
    (r"\bthis (?:proves|confirms|definitely means|means you have)\b", "unsupported_claim"),
    (r"\byou (?:don'?t|do not) need to (?:see|worry|consult)\b", "emergency_dismissal"),
]

def validate_ai_output(text: str):
    """Returns (is_safe, cleaned_text, flags). Unsafe content is rephrased conservatively."""
    flags = []
    lowered = text.lower()
    for pat, kind in _UNSAFE_PATTERNS:
        if re.search(pat, lowered):
            flags.append(kind)
    if flags:
        safe = ("I can share general educational information, but I can't provide a diagnosis, "
                "predict disease, or make medication recommendations. "
                "Recorded values and trends can be discussed with your healthcare professional, "
                "who can interpret them in your full clinical context.\n\n" + AI_DISCLAIMER)
        return False, safe, flags
    if AI_DISCLAIMER not in text:
        text = text.rstrip() + "\n\n" + AI_DISCLAIMER
    return True, text, flags


VRIKKA_SYSTEM_PROMPT = (
    "You are Vrikka AI, an educational assistant inside VrikkaPulse, a kidney-health monitoring "
    "and education companion. Your ONLY role is to explain kidney-related concepts in plain, "
    "reassuring, accessible language for patients and caregivers (including older adults).\n\n"
    "You may explain: eGFR, creatinine, UACR/ACR, albuminuria, CKD CGA categories (G1–G5, A1–A3), "
    "cystatin C, kidney terminology, report terminology, general kidney-health concepts, and helpful "
    "questions a person could ask their doctor.\n\n"
    "STRICT RULES — you MUST NOT:\n"
    "- Diagnose any condition (CKD, AKI, kidney failure) or say the user 'has' a disease.\n"
    "- Predict future disease or kidney failure.\n"
    "- Prescribe, recommend starting/stopping/changing/replacing any medication or dose.\n"
    "- Say a user is 'safe', 'healthy', or that kidneys are 'fine/failing'.\n"
    "- Invent clinical scores, formulas, guidelines, or reference ranges.\n"
    "- Give false reassurance or dismiss serious symptoms.\n"
    "- Interpret the user's specific results as proof of a condition.\n\n"
    "Always frame things as general education. Use phrases like 'recorded value', 'in general', "
    "'this typically means', and 'discuss with your healthcare professional'. "
    "When relevant, cite KDIGO 2024 as the guideline. Keep answers concise and kind. "
    "Always defer specific interpretation and decisions to the user's healthcare professional."
)
