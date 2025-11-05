# Explainability in Absenteeism Prediction: Global and Local Insights

## 1. Introduction

This report presents a comprehensive explainability analysis of the absenteeism prediction model, a linear regression system trained on workplace and employee attributes to estimate absenteeism hours. Explainability is crucial for understanding model decisions, ensuring transparency, and enabling responsible AI deployment. We employ both global and local explanation techniques to answer critical user questions about model behavior.

## 2. Techniques Used

### 2.1 Global Explainability

**Linear Model Coefficients**: For linear regression models, coefficients directly indicate feature importance. We computed absolute coefficient values to rank features by their global influence on predictions. The analysis revealed that Body mass index (coefficient: -14.52) and Weight (coefficient: 12.54) are the strongest global drivers, with negative coefficients indicating decreased absenteeism prediction and positive coefficients indicating increased prediction.

**SHAP Global Values**: SHAP (SHapley Additive exPlanations) values provide a unified framework for explaining model predictions by quantifying each feature's contribution based on cooperative game theory. We computed mean absolute SHAP values across 50 training samples to identify globally important features. The results align with coefficient analysis: Body mass index (mean absolute SHAP: 11.72) and Weight (10.71) remain the top contributors, confirming the robustness of these findings across different explanation methods.

### 2.2 Local Explainability

**LIME (Local Interpretable Model-agnostic Explanations)**: LIME explains individual predictions by training a simple, interpretable model (linear) in the neighborhood of the instance. For each test instance, LIME samples nearby points, gets predictions from the black-box model, and learns which features matter locally. This provides intuitive, instance-specific explanations that highlight which feature ranges (e.g., "Weight <= 69.00") drive the prediction.

**SHAP Instance-Level Explanations**: SHAP provides local explanations through Shapley values, which fairly distribute the prediction among features. We generated waterfall plots and force plots showing how each feature pushes the prediction up or down from the baseline (expected value). SHAP values are additive: the sum of all feature contributions plus the baseline equals the prediction, ensuring mathematical consistency.

## 3. Why These Methods Were Chosen

**Interpretability**: Linear regression coefficients are inherently interpretable—each coefficient represents the expected change in prediction per unit change in the feature. This makes them ideal for global understanding, especially for non-technical stakeholders who can directly interpret "higher BMI decreases predicted absenteeism by 14.52 hours."

**Model-Agnostic Approach**: Both LIME and SHAP are model-agnostic, meaning they work with any black-box model (not just linear models). This flexibility allows us to explain the pipeline (which includes scaling) as a whole, rather than just the linear regression component.

**Suited for Regression**: Unlike classification explanations that focus on class probabilities, these methods handle continuous outputs (absenteeism hours) effectively. SHAP values are particularly well-suited for regression as they preserve the additive property: `prediction = baseline + Σ(SHAP_value_i)`, making it easy to understand how features combine to produce the final prediction.

**Complementary Strengths**: LIME provides intuitive, rule-based explanations ("Weight <= 69.00 decreases prediction"), while SHAP offers mathematically rigorous, consistent contributions. Using both methods validates findings and provides multiple perspectives on the same prediction.

## 4. Addressing User Questions

### 4.1 WHY: Why Did This Instance Get This Prediction?

For a test instance (ID: 0) with predicted absenteeism of -2.04 hours, both LIME and SHAP identified the same primary drivers:

**LIME Explanation**: The prediction was primarily driven by "Weight <= 69.00" (contribution: -22.44 hours), followed by "Disciplinary failure <= 0.00" (+18.25 hours) and "Body mass index in range 23-25" (+8.26 hours). These categorical rules make it immediately clear why the model predicted low absenteeism: the employee has low weight and no disciplinary issues.

**SHAP Explanation**: Weight contributed -9.55 hours (largest negative contribution), while Body mass index contributed +8.57 hours (largest positive contribution). The net effect of these competing forces, combined with other features (Reason for absence: -4.14, Age: -3.53), resulted in the -2.04 hour prediction.

**Key Insight**: Both methods agree that Weight is the dominant factor, but they present it differently—LIME as a threshold rule, SHAP as a continuous contribution. This dual perspective helps users understand both the "what" (which features matter) and the "how much" (quantitative impact).

### 4.2 WHY NOT: Why Didn't It Predict Differently?

The model didn't predict a higher value (e.g., positive absenteeism hours) because several features pulled the prediction downward. Specifically:

- **Weight (≤69 kg)**: This feature had the strongest downward pull (-22.44 hours in LIME, -9.55 in SHAP). The employee's low weight is associated with lower absenteeism in the training data.

- **Reason for absence**: With a value of 27.0, this feature contributed -4.14 hours (SHAP), further reducing the prediction.

- **Age**: At 28 years, this feature contributed -3.53 hours, indicating younger employees in this dataset tend to have lower absenteeism.

Conversely, the model didn't predict an even lower value because:
- **Disciplinary failure = 0**: This increased the prediction by +18.25 hours (LIME), as employees without disciplinary issues may have different absence patterns.
- **Body mass index (24.0)**: This pushed the prediction up by +8.57 hours (SHAP), partially offsetting the weight effect.

**The net result**: The final prediction (-2.04 hours) is the algebraic sum of these competing forces. The model's prediction reflects the balance between factors increasing and decreasing absenteeism, not a single dominant feature.

### 4.3 WHAT IF: What Happens If I Change an Input Feature?

Based on the model's coefficients, we can answer counterfactual questions:

**What if Weight increased by 10 kg?** The coefficient for Weight is +12.54, so increasing weight by 10 kg would increase the prediction by approximately 12.54 × 10 = 125.4 hours. However, this is a linear approximation; in reality, the relationship may be non-linear, and the model would need retraining to capture such changes accurately.

**What if Body mass index increased by 5 units?** With a coefficient of -14.52, increasing BMI by 5 would decrease the prediction by approximately 72.6 hours. This counterintuitive result (higher BMI decreases absenteeism) suggests the model may have learned spurious correlations or the relationship is more complex than linear.

**What if Age increased by 10 years?** The coefficient for Age is approximately +2.41 (from SHAP global importance), so increasing age by 10 years would increase the prediction by about 24.1 hours.

**Limitation**: These "what-if" scenarios assume linear relationships hold outside the training data range. For non-linear effects or interactions, feature changes may have unexpected impacts.

### 4.4 HOW TO BE THAT / HOW TO REMAIN HERE: What Changes Keep or Change Classification?

While this is a regression task (predicting hours, not classes), we can frame this question in terms of prediction thresholds:

**How to remain in low absenteeism range (e.g., < 5 hours)?**
- **Maintain low Weight**: Keep weight ≤ 69 kg (strongest protective factor: -22.44 hours contribution)
- **Avoid disciplinary issues**: Maintain disciplinary failure = 0 (though this increases prediction by +18.25 hours in this instance, suggesting complex interactions)
- **Keep Body mass index in moderate range**: BMI 23-25 increases prediction by +8.26 hours, but extreme values may have different effects
- **Maintain current Age**: Younger age (28 years) contributes -3.53 hours

**How to move to high absenteeism range (e.g., > 20 hours)?**
- **Increase Weight**: Higher weight is associated with increased absenteeism (coefficient: +12.54)
- **Change Reason for absence**: Different absence reasons have varying impacts; value 27.0 currently reduces prediction
- **Increase Age**: Older employees show higher absenteeism in the model (positive coefficient)

**Critical Note**: These recommendations are based on statistical associations, not causal relationships. Changing weight or age to manipulate predictions would be unethical and may not reflect true causal effects. The model captures correlations in historical data, which may include biases or spurious patterns.

## 5. Transparency and Responsible AI

Explainability techniques like SHAP and LIME are essential for transparency and responsible AI deployment, especially for non-ML users. When HR managers or operations staff use the absenteeism prediction model, they need to understand not just what the model predicts, but why it made that prediction. Global explanations reveal that the model heavily relies on Body mass index and Weight—features that may raise fairness concerns if they lead to discriminatory practices. Local explanations help users verify that individual predictions are reasonable: if an employee receives a high absenteeism prediction, managers can see exactly which attributes drove that result and assess whether the reasoning is fair and legitimate. This transparency enables users to challenge problematic predictions, identify potential biases (e.g., age-based discrimination), and make informed decisions about when to trust or override model recommendations. Without explainability, the model remains a "black box" that users must accept on faith, which is unacceptable for decisions affecting employee welfare. By providing clear, interpretable explanations, we empower users to use the model responsibly, recognize its limitations, and maintain human oversight over automated decisions.

## 6. Comparison: LIME vs SHAP

| Aspect | LIME | SHAP |
|--------|------|------|
| **Purpose** | Provides local, interpretable explanations by training a simple model in the instance neighborhood | Provides mathematically rigorous Shapley values that fairly distribute prediction contributions |
| **Scope** | Instance-specific; explains one prediction at a time | Can be computed globally (mean SHAP) or locally (instance SHAP) |
| **Interpretability** | Very high—uses simple rules (e.g., "Weight <= 69.00") that non-technical users can understand | High—uses numerical contributions that require some interpretation but are mathematically consistent |
| **Mathematical Properties** | No guarantee of additivity or consistency across instances | Additive (prediction = baseline + Σ SHAP), satisfies efficiency, symmetry, and dummy properties |
| **Computational Cost** | Moderate—requires sampling nearby instances and training local models | Higher for exact computation; approximations (e.g., LinearExplainer) are faster |
| **Model Assumptions** | Assumes local linearity holds in the neighborhood | Makes minimal assumptions; works with any model |
| **Limitations** | Explanations may vary with different random seeds; no global consistency guarantee | Can be computationally expensive for large datasets; requires background data |
| **Best For** | Quick, intuitive understanding of why a specific prediction was made | Rigorous analysis requiring consistency and mathematical guarantees |

**Conclusion**: LIME excels at providing intuitive, rule-based explanations that answer "why this prediction?" in plain language. SHAP excels at providing mathematically rigorous, consistent explanations that can be aggregated globally or analyzed locally. Using both methods provides complementary insights: LIME's simplicity helps users understand, while SHAP's rigor ensures correctness.

## 7. Summary

This explainability analysis demonstrates that the absenteeism prediction model is driven primarily by Body mass index and Weight globally, with instance-specific predictions influenced by Weight thresholds, disciplinary status, and age. The combination of coefficient analysis, SHAP, and LIME provides multiple perspectives on model behavior, enabling users to understand both global patterns and local reasoning. These explanations support transparency, help identify potential biases, and empower users to make informed, responsible decisions when using the model for workplace planning and resource allocation.

---

*Report generated from explainability analysis of the absenteeism prediction model. All visualizations and detailed findings are available in the `explanations/` directory.*

