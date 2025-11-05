import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib, json, numpy as np

# Load data: pick only the CSV inside the ZIP
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip"
response = requests.get(url, timeout=60)
response.raise_for_status()
with ZipFile(BytesIO(response.content)) as zf:
    with zf.open("Absenteeism_at_work.csv") as csv_file:
        df = pd.read_csv(csv_file, sep=';').dropna()

target = "Absenteeism time in hours"
X = df.drop(columns=[target])
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
model.fit(X_train, y_train)

# Predict and evaluate baseline model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Fairness check: Compare performance by 'Age' groups
if "Age" in X_test.columns:
    X_test_copy = X_test.copy()
    X_test_copy["pred"] = y_pred
    X_test_copy["true"] = y_test.values
    X_test_copy["error"] = abs(X_test_copy["pred"] - X_test_copy["true"])
    fairness_baseline = X_test_copy.groupby(pd.cut(X_test_copy["Age"], bins=[18,30,40,50,60]))["error"].mean().to_dict()
else:
    fairness_baseline = {"Fairness metric": "Age attribute missing"}

# Save baseline model
joblib.dump(model, "absenteeism_model.pkl")

# ===== FAIRNESS MITIGATION: Reweighting by Age Groups =====
print("Training fairness-mitigated model using reweighting...")

# Create age groups for training data
if "Age" in X_train.columns:
    age_groups_train = pd.cut(X_train["Age"], bins=[18,30,40,50,60], labels=["18-30", "30-40", "40-50", "50-60"])
    
    # Calculate inverse frequency weights to balance groups
    group_counts = age_groups_train.value_counts()
    total_samples = len(X_train)
    num_groups = len(group_counts)
    weights_dict = {}
    for group in group_counts.index:
        # Inverse frequency weighting: less frequent groups get higher weights
        weights_dict[group] = total_samples / (num_groups * group_counts[group])
    
    # Create sample weights array
    sample_weights = np.array([weights_dict.get(age_groups_train.iloc[i], 1.0) for i in range(len(X_train))])
    
    # Train mitigated model with sample weights
    # Note: We need to manually scale and fit to use sample_weight
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model_mitigated_lr = LinearRegression()
    model_mitigated_lr.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # Create a pipeline for consistency (but we'll use it differently)
    model_mitigated = Pipeline([
        ("scaler", scaler),
        ("lr", model_mitigated_lr)
    ])
    
    # Evaluate mitigated model
    y_pred_mitigated = model_mitigated.predict(X_test)
    mae_mitigated = mean_absolute_error(y_test, y_pred_mitigated)
    r2_mitigated = r2_score(y_test, y_pred_mitigated)
    
    # Fairness check for mitigated model
    X_test_mit = X_test.copy()
    X_test_mit["pred"] = y_pred_mitigated
    X_test_mit["true"] = y_test.values
    X_test_mit["error"] = abs(X_test_mit["pred"] - X_test_mit["true"])
    fairness_mitigated = X_test_mit.groupby(pd.cut(X_test_mit["Age"], bins=[18,30,40,50,60]))["error"].mean().to_dict()
    
    # Calculate fairness improvement (reduction in max error disparity)
    if fairness_baseline and fairness_mitigated:
        baseline_errors = [v for v in fairness_baseline.values() if isinstance(v, (int, float))]
        mitigated_errors = [v for k, v in fairness_mitigated.items() if isinstance(v, (int, float))]
        if baseline_errors and mitigated_errors:
            max_baseline = max(baseline_errors)
            min_baseline = min(baseline_errors)
            max_mitigated = max(mitigated_errors)
            min_mitigated = min(mitigated_errors)
            disparity_baseline = max_baseline - min_baseline
            disparity_mitigated = max_mitigated - min_mitigated
            improvement = disparity_baseline - disparity_mitigated
            improvement_pct = (improvement / disparity_baseline * 100) if disparity_baseline > 0 else 0
        else:
            improvement = 0
            improvement_pct = 0
    else:
        improvement = 0
        improvement_pct = 0
    
    # Save mitigated model
    joblib.dump(model_mitigated, "absenteeism_model_mitigated.pkl")
    
    print(f"✅ Mitigated model trained. Fairness improvement: {improvement:.3f} ({improvement_pct:.1f}% reduction in error disparity)")
else:
    model_mitigated = model
    mae_mitigated = mae
    r2_mitigated = r2
    fairness_mitigated = fairness_baseline
    improvement = 0
    improvement_pct = 0

# Save training feature schema and means for imputing missing values at inference
feature_columns = list(X.columns)
feature_means = {col: float(X[col].mean()) for col in feature_columns}
json.dump(feature_columns, open("feature_columns.json", "w"))
json.dump(feature_means, open("feature_means.json", "w"))

# Save metrics with both baseline and mitigated results
info = {
    "baseline": {
        "r2_score": round(r2, 3),
        "mae": round(mae, 3),
        "fairness_error_by_age": {str(k): round(v, 3) for k, v in fairness_baseline.items()}
    },
    "mitigated": {
        "r2_score": round(r2_mitigated, 3),
        "mae": round(mae_mitigated, 3),
        "fairness_error_by_age": {str(k): round(v, 3) for k, v in fairness_mitigated.items()},
        "improvement": {
            "error_disparity_reduction": round(improvement, 3),
            "improvement_percentage": round(improvement_pct, 1)
        }
    },
    "limitations": [
        "Predictions are statistical estimates, not exact outcomes.",
        "Trained only on historical data from one workplace; may not generalize.",
        "Linear regression assumes linear relationships."
    ],
    "mitigation_applied": True,
    "mitigation_strategy": "Inverse frequency reweighting by age groups"
}
json.dump(info, open("model_info.json", "w"), indent=4)

print("✅ Models trained and saved (baseline + mitigated).")
