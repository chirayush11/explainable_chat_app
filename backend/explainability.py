"""
Explainability Analysis for Absenteeism Prediction Model

This script provides global and local explanations for the absenteeism prediction model.
It answers user questions like WHY, WHY NOT, and WHAT IF by analyzing:
- Global feature importance (coefficients)
- SHAP values for global understanding
- LIME and SHAP for local instance explanations
"""

import pandas as pd
import numpy as np
import requests
from io import BytesIO
from zipfile import ZipFile
import joblib
import json
import os
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# Create explanations directory if it doesn't exist
os.makedirs("explanations", exist_ok=True)

print("=" * 60)
print("Absenteeism Model Explainability Analysis")
print("=" * 60)

# ============================================================================
# 1. LOAD DATA AND MODEL
# ============================================================================

print("\n[1/6] Loading dataset and trained model...")

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip"
response = requests.get(url, timeout=60)
response.raise_for_status()
with ZipFile(BytesIO(response.content)) as zf:
    with zf.open("Absenteeism_at_work.csv") as csv_file:
        df = pd.read_csv(csv_file, sep=';').dropna()

target = "Absenteeism time in hours"
X = df.drop(columns=[target])
y = df[target]

# Load trained model
model = joblib.load("absenteeism_model.pkl")
print(f"✓ Model loaded: {type(model).__name__}")
print(f"✓ Dataset shape: {X.shape}")
print(f"✓ Features: {list(X.columns)}")

# Prepare data for explainability (use training data for SHAP background)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get the actual model from pipeline for coefficient access
if hasattr(model, 'named_steps'):
    model_lr = model.named_steps['lr']
    scaler = model.named_steps['scaler']
    X_train_scaled = scaler.transform(X_train)
    feature_names = X.columns.tolist()
else:
    model_lr = model
    scaler = None
    X_train_scaled = X_train.values
    feature_names = X.columns.tolist()

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# ============================================================================
# 2. GLOBAL EXPLANATIONS: Feature Importance (Coefficients)
# ============================================================================

print("\n[2/6] Computing global feature importance (coefficients)...")

# Get coefficients from linear regression
coefficients = model_lr.coef_
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

# Top features
top_features = feature_importance.head(10)

print("\nTop 10 Features by Absolute Coefficient:")
print(top_features[['Feature', 'Coefficient', 'Abs_Coefficient']].to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 8))
colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 10 Feature Importance (Linear Regression Coefficients)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('explanations/global_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: explanations/global_feature_importance.png")
plt.close()

# ============================================================================
# 3. GLOBAL EXPLANATIONS: SHAP Values
# ============================================================================

print("\n[3/6] Computing global SHAP values...")

# Use a sample of training data as background for SHAP (faster computation)
background_sample = X_train_scaled[:100]  # Sample 100 instances for speed

# Create SHAP explainer for linear model
explainer_shap = shap.LinearExplainer(model_lr, background_sample)
shap_values = explainer_shap.shap_values(X_train_scaled[:50])  # Explain first 50 samples

# Mean absolute SHAP values for global importance
mean_abs_shap = np.abs(shap_values).mean(0)
shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("\nTop 10 Features by Mean Absolute SHAP Value:")
print(shap_importance.head(10).to_string(index=False))

# Visualize SHAP summary
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train_scaled[:50], feature_names=feature_names, 
                  show=False, plot_type="bar")
plt.tight_layout()
plt.savefig('explanations/shap_global_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: explanations/shap_global_summary.png")
plt.close()

# SHAP summary plot (beeswarm)
try:
    shap.summary_plot(shap_values, X_train_scaled[:50], feature_names=feature_names, 
                     show=False)
    plt.tight_layout()
    plt.savefig('explanations/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: explanations/shap_summary_beeswarm.png")
    plt.close()
except Exception as e:
    print(f"Note: Could not create beeswarm plot: {e}")

# ============================================================================
# 4. LOCAL EXPLANATIONS: LIME for Individual Predictions
# ============================================================================

print("\n[4/6] Computing local explanations with LIME...")

# Prepare LIME explainer
# LIME works with raw features (before scaling)
explainer_lime = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Absenteeism Hours'],
    mode='regression'
)

# Select a few interesting instances for explanation
sample_indices = [0, 10, 20]  # First, 11th, 21st test samples
lime_explanations = []

for idx in sample_indices:
    if idx >= len(X_test):
        continue
    
    instance = X_test.iloc[idx]
    # Model.predict handles scaling internally if it's a Pipeline
    prediction = model.predict(instance.values.reshape(1, -1))[0]
    actual = y_test.iloc[idx]
    
    # Get LIME explanation
    # LIME explainer works with raw features, model.predict handles scaling internally
    explanation = explainer_lime.explain_instance(
        instance.values,
        lambda x: model.predict(x).flatten(),
        num_features=10,
        top_labels=1
    )
    
    # Extract feature contributions
    exp_list = explanation.as_list()
    lime_explanations.append({
        'instance_idx': idx,
        'prediction': prediction,
        'actual': actual,
        'features': exp_list
    })
    
    # Visualize LIME explanation
    fig = explanation.as_pyplot_figure()
    plt.title(f'LIME Explanation for Instance {idx}\nPredicted: {prediction:.2f} hours | Actual: {actual:.2f} hours', 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'explanations/lime_instance_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: explanations/lime_instance_{idx}.png")

# ============================================================================
# 5. LOCAL EXPLANATIONS: SHAP for Individual Predictions
# ============================================================================

print("\n[5/6] Computing local SHAP explanations...")

# Select one instance for detailed SHAP explanation
instance_idx = 0
instance = X_test.iloc[instance_idx]
# For SHAP, we need scaled features since explainer expects scaled input
instance_scaled = scaler.transform(instance.values.reshape(1, -1)) if scaler else instance.values.reshape(1, -1)
prediction = model.predict(instance.values.reshape(1, -1))[0]  # Model handles scaling
actual = y_test.iloc[instance_idx]

# Compute SHAP values for this instance (SHAP explainer expects scaled features)
shap_values_instance = explainer_shap.shap_values(instance_scaled[0])

# Create waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_instance,
        base_values=explainer_shap.expected_value,
        data=instance.values,
        feature_names=feature_names
    ),
    show=False
)
plt.title(f'SHAP Waterfall Plot for Instance {instance_idx}\nPredicted: {prediction:.2f} hours | Actual: {actual:.2f} hours',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('explanations/shap_waterfall_instance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: explanations/shap_waterfall_instance.png")
plt.close()

# Create force plot (HTML)
try:
    shap.force_plot(
        explainer_shap.expected_value,
        shap_values_instance,
        instance.values,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig('explanations/shap_force_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: explanations/shap_force_plot.png")
    plt.close()
except Exception as e:
    print(f"Note: Could not create force plot: {e}")

# Create detailed SHAP bar plot for this instance
feature_contributions = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Value': shap_values_instance,
    'Abs_SHAP': np.abs(shap_values_instance)
}).sort_values('Abs_SHAP', ascending=False)

plt.figure(figsize=(10, 8))
top_contrib = feature_contributions.head(15)
colors = ['green' if v > 0 else 'red' for v in top_contrib['SHAP_Value']]
plt.barh(range(len(top_contrib)), top_contrib['SHAP_Value'], color=colors, alpha=0.7)
plt.yticks(range(len(top_contrib)), top_contrib['Feature'])
plt.xlabel('SHAP Value (contribution to prediction)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title(f'SHAP Feature Contributions for Instance {instance_idx}\n(Predicted: {prediction:.2f} hours)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('explanations/shap_local_instance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: explanations/shap_local_instance.png")
plt.close()

# ============================================================================
# 6. GENERATE REPORT
# ============================================================================

print("\n[6/6] Generating explanation report...")

# Select one instance for detailed report
report_instance = lime_explanations[0]
instance_data = X_test.iloc[report_instance['instance_idx']]

# Get top contributing features for this instance
top_lime_features = sorted(report_instance['features'], key=lambda x: abs(x[1]), reverse=True)[:5]

# Get SHAP values for this instance
instance_scaled_report = scaler.transform(instance_data.values.reshape(1, -1)) if scaler else instance_data.values.reshape(1, -1)
shap_vals_report = explainer_shap.shap_values(instance_scaled_report[0])
shap_contrib_report = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Value': shap_vals_report
}).sort_values('SHAP_Value', key=abs, ascending=False).head(5)

# Write report
report_text = f"""
{'=' * 80}
ABSENTEEISM PREDICTION MODEL - EXPLAINABILITY REPORT
{'=' * 80}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

This report explains what the model learned globally and why it made specific 
predictions for individual instances. It addresses user questions:
- WHY: Why did the model predict this value?
- WHY NOT: Why didn't it predict a different value?
- WHAT IF: What would happen if feature values changed?

{'=' * 80}
1. GLOBAL MODEL UNDERSTANDING
{'=' * 80}

The model learned patterns from {len(X_train)} training examples to predict 
absenteeism hours based on workplace and employee attributes.

TOP 5 GLOBAL DRIVERS (Features with highest absolute coefficients):

"""

for i, row in feature_importance.head(5).iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    report_text += f"{i+1}. {row['Feature']}\n"
    report_text += f"   - Coefficient: {row['Coefficient']:.4f}\n"
    report_text += f"   - Impact: Higher values of this feature {direction} predicted absenteeism\n"
    report_text += f"   - Strength: {row['Abs_Coefficient']:.4f} (absolute coefficient)\n\n"

report_text += f"""
TOP 5 GLOBAL DRIVERS (by SHAP importance):

"""

for i, row in shap_importance.head(5).iterrows():
    report_text += f"{i+1}. {row['Feature']}\n"
    report_text += f"   - Mean Absolute SHAP: {row['Mean_Abs_SHAP']:.4f}\n"
    report_text += f"   - This feature consistently contributes to predictions across all samples\n\n"

report_text += f"""
{'=' * 80}
2. LOCAL EXPLANATION: Why This Specific Prediction?
{'=' * 80}

Instance Details:
- Index: {report_instance['instance_idx']}
- Predicted Absenteeism: {report_instance['prediction']:.2f} hours
- Actual Absenteeism: {report_instance['actual']:.2f} hours
- Error: {abs(report_instance['prediction'] - report_instance['actual']):.2f} hours

Feature Values for This Instance:
"""

for feat in feature_names[:10]:  # Show first 10 features
    val = instance_data[feat]
    report_text += f"  - {feat}: {val}\n"

report_text += f"""
WHY did the model predict {report_instance['prediction']:.2f} hours?

Top 5 Features Driving This Prediction (LIME explanation):

"""

for i, (feat, contrib) in enumerate(top_lime_features, 1):
    direction = "increases" if contrib > 0 else "decreases"
    report_text += f"{i}. {feat}\n"
    report_text += f"   - Contribution: {contrib:.4f}\n"
    report_text += f"   - This feature {direction} the prediction by {abs(contrib):.4f} hours\n\n"

report_text += f"""
Top 5 Features Driving This Prediction (SHAP explanation):

"""

for i, row in shap_contrib_report.iterrows():
    direction = "increases" if row['SHAP_Value'] > 0 else "decreases"
    report_text += f"{i+1}. {row['Feature']}\n"
    report_text += f"   - SHAP Value: {row['SHAP_Value']:.4f}\n"
    report_text += f"   - This feature {direction} the prediction by {abs(row['SHAP_Value']):.4f} hours\n\n"

report_text += f"""
{'=' * 80}
3. ANSWERING USER QUESTIONS
{'=' * 80}

WHY did the model predict {report_instance['prediction']:.2f} hours?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The prediction was primarily driven by:
"""

# Combine LIME and SHAP insights
top_drivers = {}
for feat, contrib in top_lime_features:
    top_drivers[feat] = contrib

for _, row in shap_contrib_report.iterrows():
    if row['Feature'] in top_drivers:
        top_drivers[row['Feature']] = (top_drivers[row['Feature']] + row['SHAP_Value']) / 2
    else:
        top_drivers[row['Feature']] = row['SHAP_Value']

sorted_drivers = sorted(top_drivers.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

for feat, contrib in sorted_drivers:
    direction = "increased" if contrib > 0 else "decreased"
    val = instance_data[feat] if feat in instance_data.index else "N/A"
    report_text += f"  • {feat} (value: {val}) {direction} the prediction by {abs(contrib):.4f} hours\n"

report_text += f"""

WHY NOT a different value?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The model didn't predict a higher/lower value because:
"""

# Identify features that pulled the prediction in different directions
positive_features = [f for f, c in sorted_drivers if c > 0]
negative_features = [f for f, c in sorted_drivers if c < 0]

if positive_features and negative_features:
    report_text += f"  • Some features pushed the prediction UP: {', '.join(positive_features[:3])}\n"
    report_text += f"  • Other features pulled it DOWN: {', '.join(negative_features[:3])}\n"
    report_text += f"  • The final prediction ({report_instance['prediction']:.2f} hours) is the net result\n"

report_text += f"""

WHAT IF we changed feature values?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Based on the model coefficients, changing features would have these effects:
"""

# Show what-if scenarios
for feat, coef in feature_importance.head(5).iterrows():
    current_val = instance_data[feat] if feat in instance_data.index else "N/A"
    if isinstance(current_val, (int, float)):
        change_effect = coef['Coefficient'] * 1.0  # Effect of increasing by 1 unit
        report_text += f"  • If {feat} increased by 1 unit (from {current_val:.2f}): "
        report_text += f"prediction would change by {change_effect:+.4f} hours\n"

report_text += f"""
{'=' * 80}
4. KEY INSIGHTS FOR USERS
{'=' * 80}

GLOBAL INSIGHTS:
• The model identifies {feature_importance.iloc[0]['Feature']} as the most important 
  global driver of absenteeism predictions.
• The top 5 features together explain a significant portion of the model's decisions.
• Features with positive coefficients increase predicted absenteeism, while negative 
  coefficients decrease it.

LOCAL INSIGHTS (for this instance):
• The prediction of {report_instance['prediction']:.2f} hours was primarily driven by 
  {sorted_drivers[0][0]} ({sorted_drivers[0][1]:+.4f} hours contribution).
• The model's reasoning aligns with the feature values: this instance has specific 
  characteristics that led to this prediction.
• The error of {abs(report_instance['prediction'] - report_instance['actual']):.2f} hours 
  suggests the model is {'reasonably accurate' if abs(report_instance['prediction'] - report_instance['actual']) < 5 else 'has room for improvement'} for this type of case.

RECOMMENDATIONS:
• Use these explanations to understand model trustworthiness.
• Consider the top features when making decisions about workplace policies.
• Remember: predictions are estimates, not certainties.
• Check fairness metrics to ensure the model treats all groups fairly.

{'=' * 80}
END OF REPORT
{'=' * 80}
"""

# Save report
with open('explanations/report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("✓ Saved: explanations/report.txt")

# Print summary to console
print("\n" + "=" * 60)
print("EXPLANABILITY ANALYSIS COMPLETE")
print("=" * 60)
print("\n📊 SUMMARY:")
print(f"\nTop 5 Global Drivers:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {i+1}. {row['Feature']} (coef: {row['Coefficient']:.4f})")

print(f"\n📍 Local Explanation for Instance {report_instance['instance_idx']}:")
print(f"  Predicted: {report_instance['prediction']:.2f} hours")
print(f"  Actual: {report_instance['actual']:.2f} hours")
print(f"\n  Top 3 contributing features:")
for i, (feat, contrib) in enumerate(sorted_drivers[:3], 1):
    print(f"    {i}. {feat}: {contrib:+.4f} hours")

print("\n📁 All outputs saved in explanations/ folder:")
print("   - global_feature_importance.png")
print("   - shap_global_summary.png")
print("   - lime_instance_*.png")
print("   - shap_waterfall_instance.png")
print("   - shap_local_instance.png")
print("   - report.txt")

print("\n✅ Analysis complete!")

