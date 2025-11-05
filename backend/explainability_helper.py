"""
Explainability Helper Module for Real-time Explanations

Provides functions to generate local explanations for user inputs.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# Global variables to cache loaded data
_model = None
_scaler = None
_X_train = None
_feature_names = None
_lime_explainer = None
_shap_explainer = None
_feature_means = None

def _load_resources():
    """Load model and training data once, cache for reuse"""
    global _model, _scaler, _X_train, _feature_names, _lime_explainer, _shap_explainer, _feature_means
    
    if _model is not None:
        return  # Already loaded
    
    # Load model
    _model = joblib.load("absenteeism_model.pkl")
    
    # Load feature schema
    with open("feature_columns.json", 'r') as f:
        _feature_names = json.load(f)
    with open("feature_means.json", 'r') as f:
        _feature_means = json.load(f)
    
    # Load training data for explainers
    from sklearn.model_selection import train_test_split
    import requests
    from io import BytesIO
    from zipfile import ZipFile
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as zf:
        with zf.open("Absenteeism_at_work.csv") as csv_file:
            df = pd.read_csv(csv_file, sep=';').dropna()
    
    target = "Absenteeism time in hours"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    _X_train = X_train
    
    # Extract scaler from pipeline
    if hasattr(_model, 'named_steps'):
        _scaler = _model.named_steps.get('scaler')
    else:
        _scaler = None
    
    # Initialize LIME explainer
    _lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=_feature_names,
        class_names=['Absenteeism Hours'],
        mode='regression'
    )
    
    # Initialize SHAP explainer (use sample for speed)
    X_train_scaled = _scaler.transform(X_train) if _scaler else X_train.values
    background_sample = X_train_scaled[:50]  # Small sample for speed
    if hasattr(_model, 'named_steps'):
        model_lr = _model.named_steps['lr']
    else:
        model_lr = _model
    _shap_explainer = shap.LinearExplainer(model_lr, background_sample)
    
    print("✓ Explainability resources loaded")

def get_local_explanation(features_dict):
    """
    Generate local explanations (LIME and SHAP) for a given feature set.
    
    Args:
        features_dict: Dictionary of feature values (can be partial)
    
    Returns:
        Dictionary with LIME and SHAP explanations
    """
    _load_resources()
    
    # Build full feature vector
    full_features = {}
    for col in _feature_names:
        if col in features_dict and features_dict[col] is not None:
            full_features[col] = features_dict[col]
        else:
            full_features[col] = _feature_means.get(col, 0.0)
    
    # Convert to array
    instance_array = np.array([full_features[col] for col in _feature_names]).reshape(1, -1)
    instance_df = pd.DataFrame(instance_array, columns=_feature_names)
    
    # Get prediction
    prediction = float(_model.predict(instance_array)[0])
    if prediction < 0:
        prediction = 0.0
    
    # LIME explanation
    lime_exp = _lime_explainer.explain_instance(
        instance_df.iloc[0].values,
        lambda x: _model.predict(x).flatten(),
        num_features=10,
        top_labels=1
    )
    lime_list = lime_exp.as_list()
    
    # SHAP explanation
    instance_scaled = _scaler.transform(instance_array) if _scaler else instance_array
    shap_values = _shap_explainer.shap_values(instance_scaled[0])
    expected_value = _shap_explainer.expected_value
    
    # Get top contributing features
    shap_contributions = []
    for i, (feat, val) in enumerate(zip(_feature_names, shap_values)):
        shap_contributions.append({
            'feature': feat,
            'value': instance_df.iloc[0][feat],
            'shap_value': float(val),
            'abs_shap': float(np.abs(val))
        })
    shap_contributions.sort(key=lambda x: x['abs_shap'], reverse=True)
    
    # Create LIME visualization as base64
    lime_img_base64 = _create_lime_image(lime_exp, prediction)
    
    # Create SHAP visualization as base64
    shap_img_base64 = _create_shap_image(shap_values, instance_df.iloc[0].values, _feature_names, expected_value, prediction)
    
    return {
        'prediction': prediction,
        'lime': {
            'features': lime_list[:10],  # Top 10
            'image': lime_img_base64
        },
        'shap': {
            'expected_value': float(expected_value),
            'contributions': shap_contributions[:10],  # Top 10
            'image': shap_img_base64
        },
        'instance_features': full_features
    }

def _create_lime_image(lime_explanation, prediction):
    """Create LIME explanation plot and return as base64"""
    try:
        fig = lime_explanation.as_pyplot_figure()
        plt.title(f'LIME Explanation\nPredicted: {prediction:.2f} hours', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error creating LIME image: {e}")
        return None

def _create_shap_image(shap_values, instance_values, feature_names, expected_value, prediction):
    """Create SHAP explanation plot and return as base64"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get top 10 features by absolute SHAP value
        top_indices = np.argsort(np.abs(shap_values))[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_values = [shap_values[i] for i in top_indices]
        top_instance_vals = [instance_values[i] for i in top_indices]
        
        # Create horizontal bar plot
        colors = ['green' if v > 0 else 'red' for v in top_values]
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('SHAP Value (contribution to prediction)', fontsize=11)
        ax.set_title(f'SHAP Feature Contributions\nPredicted: {prediction:.2f} hours | Baseline: {expected_value:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error creating SHAP image: {e}")
        return None

def get_global_explanations():
    """Get global explanation images and feature importance"""
    _load_resources()
    
    images = {}
    
    # Load all available global explanation images
    image_files = {
        'feature_importance': 'explanations/global_feature_importance.png',
        'shap_global': 'explanations/shap_global_summary.png',
        'shap_beeswarm': 'explanations/shap_summary_beeswarm.png',
        'shap_local': 'explanations/shap_local_instance.png',
        'shap_waterfall': 'explanations/shap_waterfall_instance.png',
        'lime_instance_0': 'explanations/lime_instance_0.png',
        'lime_instance_10': 'explanations/lime_instance_10.png',
        'lime_instance_20': 'explanations/lime_instance_20.png'
    }
    
    for key, filepath in image_files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    images[key] = base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
    
    # Get coefficient-based feature importance
    if hasattr(_model, 'named_steps'):
        model_lr = _model.named_steps['lr']
    else:
        model_lr = _model
    
    coefficients = model_lr.coef_
    feature_importance = pd.DataFrame({
        'Feature': _feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    top_features = feature_importance.head(10).to_dict('records')
    
    return {
        'images': images,
        'top_features': top_features
    }

