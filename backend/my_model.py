import joblib, json, pandas as pd
import os

# Load models - use mitigated model if available, otherwise fall back to baseline
model_baseline = joblib.load("absenteeism_model.pkl")
if os.path.exists("absenteeism_model_mitigated.pkl"):
    model = joblib.load("absenteeism_model_mitigated.pkl")
    use_mitigated = True
else:
    model = model_baseline
    use_mitigated = False

info = json.load(open("model_info.json"))
feature_columns = json.load(open("feature_columns.json"))
feature_means = json.load(open("feature_means.json"))

def predict_absenteeism(features_dict, use_mitigated_model=True):
    # Build a full feature vector: use provided values, otherwise default to training mean
    full_features = {}
    for col in feature_columns:
        if col in features_dict and features_dict[col] is not None:
            full_features[col] = features_dict[col]
        else:
            full_features[col] = feature_means.get(col, 0.0)
    X = pd.DataFrame([full_features], columns=feature_columns)
    
    # Use mitigated model by default if available
    pred_model = model if (use_mitigated_model and use_mitigated) else model_baseline
    prediction = float(pred_model.predict(X)[0])
    # Post-process: absenteeism hours cannot be negative
    if prediction < 0:
        prediction = 0.0
    return prediction

def get_model_info():
    enriched = dict(info)
    # Attach feature schema and defaulting details for the UI
    enriched["feature_columns"] = feature_columns
    enriched["feature_means"] = feature_means
    
    # Update mitigation info based on what's actually available
    if "mitigation_applied" in info and info["mitigation_applied"]:
        enriched["mitigation"] = {
            "applied": True,
            "strategy": info.get("mitigation_strategy", "Inverse frequency reweighting by age groups"),
            "notes": "Fairness mitigation applied using reweighting. Both baseline and mitigated metrics shown below."
        }
        # Add backward compatibility for old UI code
        enriched["r2_score"] = enriched["mitigated"]["r2_score"]
        enriched["mae"] = enriched["mitigated"]["mae"]
        enriched["fairness_error_by_age"] = enriched["mitigated"]["fairness_error_by_age"]
    else:
        enriched["mitigation"] = {
            "applied": False,
            "strategy": "None",
            "notes": "No fairness mitigation applied; values shown are baseline."
        }
        # Fallback for old format
        if "baseline" in enriched:
            enriched["r2_score"] = enriched["baseline"]["r2_score"]
            enriched["mae"] = enriched["baseline"]["mae"]
            enriched["fairness_error_by_age"] = enriched["baseline"]["fairness_error_by_age"]
    
    return enriched
