"""Official column roles. No inferred treatment channels or label features."""

DYNAMIC = tuple("HR O2Sat Temp SBP MAP DBP Resp EtCO2 BaseExcess HCO3 FiO2 pH PaCO2 SaO2 AST BUN Alkalinephos Calcium Chloride Creatinine Bilirubin_direct Glucose Lactate Magnesium Phosphate Potassium Bilirubin_total TroponinI Hct Hgb PTT WBC Fibrinogen Platelets".split())
STATIC = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")
TIME = ("ICULOS",)
LABEL = ("SepsisLabel",)
HEADER = DYNAMIC + STATIC + TIME + LABEL
ROLES = {**dict.fromkeys(DYNAMIC, "dynamic_state"), **dict.fromkeys(STATIC, "static_context"), **dict.fromkeys(TIME, "time_context"), **dict.fromkeys(LABEL, "label")}


def validate_features(features):
    if not features or len(set(features)) != len(features) or not set(features) <= set(DYNAMIC):
        raise ValueError("Only unique native dynamic physiology variables may be forecast features")
