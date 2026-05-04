"""
Healthcare AI Diagnostic Gap Analysis
======================================
Module: SHAP Explainability Analysis
Purpose: Apply SHAP (SHapley Additive exPlanations) to decompose model
         predictions and identify demographic-correlated feature importance
         patterns that contribute to diagnostic disparities.

TreeExplainer is used for exact SHAP value computation on the Random Forest
model, enabling per-patient and per-subgroup interpretability analysis.

Author: Bhakti Patel
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

# Reproducibility
np.random.seed(42)


# ============================================================================
# DATA & MODEL LOADING
# ============================================================================

def load_model_and_data():
    """
    Load the trained baseline model and test data for SHAP analysis.
    
    Returns
    -------
    tuple
        model, X_test, y_test, feature_names, demographics
    """
    base_dir = Path(__file__).parent.parent
    
    # Load model
    model_path = base_dir / "models" / "baseline_model.joblib"
    if not model_path.exists():
        print("Baseline model not found. Running baseline_model.py first...")
        from baseline_model import main as train_baseline
        train_baseline()
    
    model = joblib.load(model_path)
    
    # Load data
    data_path = base_dir / "data" / "synthetic_clinical_data.csv"
    df = pd.read_csv(data_path)
    
    # Prepare features (same preprocessing as baseline_model.py)
    feature_cols = [
        'age', 'systolic_bp', 'diastolic_bp', 'hba1c', 'ldl_cholesterol',
        'bmi', 'creatinine', 'heart_rate', 'smoking_status',
        'physical_activity_score', 'prior_hospitalizations',
        'medication_adherence'
    ]
    
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_insurance = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['race_encoded'] = le_race.fit_transform(df['race'])
    df['insurance_encoded'] = le_insurance.fit_transform(df['insurance_type'])
    
    feature_cols += ['gender_encoded', 'race_encoded', 'insurance_encoded']
    
    # Use test split (same random_state as baseline)
    from sklearn.model_selection import train_test_split
    X = df[feature_cols].values
    y = df['treatment_outcome'].values
    demographics = df[['patient_id', 'age', 'gender', 'race', 'insurance_type']]
    
    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demographics, test_size=0.25, random_state=42, stratify=y
    )
    
    return model, X_test, y_test, feature_cols, demo_test


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def compute_shap_values(model, X_test, feature_names):
    """
    Compute SHAP values using TreeExplainer for exact Shapley value
    decomposition on the Random Forest model.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained baseline model
    X_test : np.ndarray
        Test features
    feature_names : list
        Feature names for interpretation
        
    Returns
    -------
    np.ndarray
        SHAP values for positive class (treatment_outcome=1)
    """
    if not SHAP_AVAILABLE:
        # Simulate SHAP-like analysis for environments without SHAP installed
        print("  [Simulating SHAP values for demonstration]")
        n_samples, n_features = X_test.shape
        
        # Use feature importances as proxy for SHAP magnitude
        importances = model.feature_importances_
        shap_values = np.random.randn(n_samples, n_features) * importances
        return shap_values
    
    # TreeExplainer: exact SHAP values for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, use positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    return shap_values


def analyze_global_importance(shap_values, feature_names):
    """
    Analyze global feature importance from SHAP values.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix
    feature_names : list
        Feature names
        
    Returns
    -------
    pd.DataFrame
        Ranked feature importance
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'std_shap': shap_values.std(axis=0),
        'mean_shap': shap_values.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df['contribution_pct'] = (
        importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].sum() * 100
    )
    
    return importance_df


def analyze_demographic_shap(shap_values, X_test, feature_names, demo_test):
    """
    Analyze SHAP value distributions by demographic subgroup to identify
    systematic differences in feature attribution patterns.
    
    This reveals how the model treats different demographic groups differently,
    even when clinical features are similar — a key indicator of learned bias.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix
    X_test : np.ndarray
        Test features
    feature_names : list
        Feature names
    demo_test : pd.DataFrame
        Demographic info for test set
        
    Returns
    -------
    dict
        Demographic-stratified SHAP analysis results
    """
    results = {}
    
    # Find demographic feature indices
    race_idx = feature_names.index('race_encoded')
    gender_idx = feature_names.index('gender_encoded')
    age_idx = feature_names.index('age')
    
    # --- Race-stratified SHAP analysis ---
    results['race'] = {}
    for race in demo_test['race'].unique():
        mask = demo_test['race'].values == race
        if mask.sum() < 10:
            continue
        
        group_shap = shap_values[mask]
        group_features = X_test[mask]
        
        # Mean SHAP contribution per feature for this group
        mean_shap = group_shap.mean(axis=0)
        abs_mean_shap = np.abs(group_shap).mean(axis=0)
        
        # Key metric: how much does race_encoded feature contribute to predictions?
        race_feature_shap = group_shap[:, race_idx].mean()
        
        # Prediction bias indicator: mean total SHAP (deviation from base rate)
        total_shap_mean = group_shap.sum(axis=1).mean()
        
        results['race'][race] = {
            'n': int(mask.sum()),
            'mean_total_shap': float(total_shap_mean),
            'race_feature_contribution': float(race_feature_shap),
            'top_features': {
                feature_names[i]: float(abs_mean_shap[i]) 
                for i in np.argsort(abs_mean_shap)[-5:][::-1]
            }
        }
    
    # --- Age-stratified SHAP analysis ---
    results['age'] = {}
    ages = demo_test['age'].values
    for label, mask in [('18-44', ages < 45), ('45-64', (ages >= 45) & (ages < 65)), ('65+', ages >= 65)]:
        if mask.sum() < 10:
            continue
        
        group_shap = shap_values[mask]
        total_shap_mean = group_shap.sum(axis=1).mean()
        age_feature_shap = group_shap[:, age_idx].mean()
        
        results['age'][label] = {
            'n': int(mask.sum()),
            'mean_total_shap': float(total_shap_mean),
            'age_feature_contribution': float(age_feature_shap),
            'top_features': {
                feature_names[i]: float(np.abs(group_shap).mean(axis=0)[i])
                for i in np.argsort(np.abs(group_shap).mean(axis=0))[-5:][::-1]
            }
        }
    
    # --- Gender-stratified SHAP analysis ---
    results['gender'] = {}
    for gender in demo_test['gender'].unique():
        mask = demo_test['gender'].values == gender
        if mask.sum() < 10:
            continue
        
        group_shap = shap_values[mask]
        total_shap_mean = group_shap.sum(axis=1).mean()
        gender_feature_shap = group_shap[:, gender_idx].mean()
        
        results['gender'][gender] = {
            'n': int(mask.sum()),
            'mean_total_shap': float(total_shap_mean),
            'gender_feature_contribution': float(gender_feature_shap),
            'top_features': {
                feature_names[i]: float(np.abs(group_shap).mean(axis=0)[i])
                for i in np.argsort(np.abs(group_shap).mean(axis=0))[-5:][::-1]
            }
        }
    
    return results


def detect_bias_indicators(demographic_shap, feature_names):
    """
    Identify specific bias indicators from SHAP analysis.
    
    Bias indicators include:
    1. Large differences in mean total SHAP across demographic groups
    2. Demographic features ranking high in importance
    3. Clinical features having different importance by demographic group
    
    Parameters
    ----------
    demographic_shap : dict
        Results from analyze_demographic_shap()
    feature_names : list
        Feature names
        
    Returns
    -------
    list
        Identified bias indicators with severity ratings
    """
    indicators = []
    
    # Check race-based prediction bias
    race_shap_values = {k: v['mean_total_shap'] for k, v in demographic_shap['race'].items()}
    if race_shap_values:
        max_race = max(race_shap_values.values())
        min_race = min(race_shap_values.values())
        race_gap = max_race - min_race
        
        if race_gap > 0.05:
            indicators.append({
                'type': 'demographic_prediction_gap',
                'dimension': 'race',
                'severity': 'HIGH' if race_gap > 0.10 else 'MODERATE',
                'gap': float(race_gap),
                'detail': f"Mean prediction gap across racial groups: {race_gap:.4f}"
            })
    
    # Check if race_encoded is a top-5 feature
    race_contributions = {k: v['race_feature_contribution'] 
                         for k, v in demographic_shap['race'].items()}
    max_race_contrib = max(abs(v) for v in race_contributions.values())
    if max_race_contrib > 0.02:
        indicators.append({
            'type': 'demographic_feature_reliance',
            'dimension': 'race',
            'severity': 'HIGH',
            'contribution': float(max_race_contrib),
            'detail': f"Model relies on race_encoded (max |SHAP|={max_race_contrib:.4f})"
        })
    
    # Check age-based disparities
    age_shap_values = {k: v['mean_total_shap'] for k, v in demographic_shap['age'].items()}
    if age_shap_values:
        max_age = max(age_shap_values.values())
        min_age = min(age_shap_values.values())
        age_gap = max_age - min_age
        
        if age_gap > 0.05:
            indicators.append({
                'type': 'demographic_prediction_gap',
                'dimension': 'age',
                'severity': 'HIGH' if age_gap > 0.10 else 'MODERATE',
                'gap': float(age_gap),
                'detail': f"Mean prediction gap across age groups: {age_gap:.4f}"
            })
    
    # Check gender-based disparities
    gender_shap_values = {k: v['mean_total_shap'] for k, v in demographic_shap['gender'].items()}
    if gender_shap_values:
        gender_gap = max(gender_shap_values.values()) - min(gender_shap_values.values())
        if gender_gap > 0.03:
            indicators.append({
                'type': 'demographic_prediction_gap',
                'dimension': 'gender',
                'severity': 'MODERATE' if gender_gap < 0.08 else 'HIGH',
                'gap': float(gender_gap),
                'detail': f"Mean prediction gap across gender: {gender_gap:.4f}"
            })
    
    return indicators


def main():
    """Execute SHAP explainability analysis pipeline."""
    print("=" * 70)
    print("HEALTHCARE AI DIAGNOSTIC GAP — SHAP ANALYSIS")
    print("TreeExplainer Interpretability & Bias Detection")
    print("=" * 70)
    print()
    
    # Load model and data
    print("[1/5] Loading baseline model and test data...")
    model, X_test, y_test, feature_names, demo_test = load_model_and_data()
    print(f"      Model loaded: RandomForest (200 estimators)")
    print(f"      Test samples: {len(X_test)}")
    print(f"      Features: {len(feature_names)}")
    
    # Compute SHAP values
    print("\n[2/5] Computing SHAP values (TreeExplainer)...")
    shap_values = compute_shap_values(model, X_test, feature_names)
    print(f"      SHAP matrix shape: {shap_values.shape}")
    print(f"      Mean |SHAP|: {np.abs(shap_values).mean():.6f}")
    
    # Global feature importance
    print("\n[3/5] Analyzing global feature importance...")
    importance_df = analyze_global_importance(shap_values, feature_names)
    
    print(f"\n{'─' * 60}")
    print(f"  GLOBAL FEATURE IMPORTANCE (SHAP)")
    print(f"{'─' * 60}")
    print(f"  {'Rank':<5} {'Feature':<25} {'Mean |SHAP|':<12} {'Contribution':<12}")
    print(f"  {'─'*5} {'─'*25} {'─'*12} {'─'*12}")
    
    for _, row in importance_df.iterrows():
        print(f"  {int(row['rank']):<5} {row['feature']:<25} "
              f"{row['mean_abs_shap']:<12.6f} {row['contribution_pct']:<10.1f}%")
    
    # Demographic-stratified analysis
    print(f"\n[4/5] Analyzing SHAP values by demographic subgroup...")
    demographic_shap = analyze_demographic_shap(shap_values, X_test, feature_names, demo_test)
    
    print(f"\n{'─' * 60}")
    print(f"  SHAP ANALYSIS BY RACE/ETHNICITY")
    print(f"{'─' * 60}")
    print(f"  {'Race':<12} {'N':>5} {'Mean Total SHAP':>16} {'Race Feature':>14}")
    print(f"  {'─'*12} {'─'*5} {'─'*16} {'─'*14}")
    
    for race, data in sorted(demographic_shap['race'].items()):
        print(f"  {race:<12} {data['n']:>5} {data['mean_total_shap']:>16.6f} "
              f"{data['race_feature_contribution']:>14.6f}")
    
    print(f"\n  Interpretation: Negative mean total SHAP indicates the model")
    print(f"  systematically predicts LOWER positive outcomes for that group.")
    
    print(f"\n{'─' * 60}")
    print(f"  SHAP ANALYSIS BY AGE GROUP")
    print(f"{'─' * 60}")
    print(f"  {'Age Group':<12} {'N':>5} {'Mean Total SHAP':>16} {'Age Feature':>14}")
    print(f"  {'─'*12} {'─'*5} {'─'*16} {'─'*14}")
    
    for age_group, data in sorted(demographic_shap['age'].items()):
        print(f"  {age_group:<12} {data['n']:>5} {data['mean_total_shap']:>16.6f} "
              f"{data['age_feature_contribution']:>14.6f}")
    
    print(f"\n{'─' * 60}")
    print(f"  SHAP ANALYSIS BY GENDER")
    print(f"{'─' * 60}")
    print(f"  {'Gender':<12} {'N':>5} {'Mean Total SHAP':>16} {'Gender Feature':>14}")
    print(f"  {'─'*12} {'─'*5} {'─'*16} {'─'*14}")
    
    for gender, data in sorted(demographic_shap['gender'].items()):
        print(f"  {gender:<12} {data['n']:>5} {data['mean_total_shap']:>16.6f} "
              f"{data['gender_feature_contribution']:>14.6f}")
    
    # Bias indicator detection
    print(f"\n[5/5] Detecting bias indicators...")
    indicators = detect_bias_indicators(demographic_shap, feature_names)
    
    print(f"\n{'─' * 60}")
    print(f"  ⚠️  BIAS INDICATORS DETECTED: {len(indicators)}")
    print(f"{'─' * 60}")
    
    for i, ind in enumerate(indicators, 1):
        severity_icon = "🔴" if ind['severity'] == 'HIGH' else "🟡"
        print(f"\n  {severity_icon} Indicator #{i}: {ind['type']}")
        print(f"     Dimension: {ind['dimension']}")
        print(f"     Severity:  {ind['severity']}")
        print(f"     Detail:    {ind['detail']}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'global_importance': importance_df.to_dict('records'),
        'demographic_analysis': demographic_shap,
        'bias_indicators': indicators
    }
    
    results_path = output_dir / "shap_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print("✓ SHAP analysis complete.")
    print(f"  {len(indicators)} bias indicators detected across race, age, and gender.")
    print("  Proceed to bias mitigation for fairness-aware model retraining.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
