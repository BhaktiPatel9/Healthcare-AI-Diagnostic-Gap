"""
Healthcare AI Diagnostic Gap Analysis
======================================
Module: Baseline Model Training
Purpose: Train a standard Random Forest classifier for diagnostic outcome
         prediction WITHOUT fairness constraints. This establishes the baseline
         performance (AUC ~0.74) that reveals demographic disparities.

The baseline model intentionally does not account for demographic bias,
allowing subsequent modules to detect and quantify the diagnostic gap.

Author: Bhakti Patel
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
import joblib

# Reproducibility
np.random.seed(42)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess():
    """
    Load synthetic clinical dataset and prepare features for model training.
    
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, feature_names, demographics_test
    """
    # Load dataset
    data_path = Path(__file__).parent.parent / "data" / "synthetic_clinical_data.csv"
    
    if not data_path.exists():
        print("Dataset not found. Running data_preparation.py first...")
        from data_preparation import main as prepare_data
        prepare_data()
    
    df = pd.read_csv(data_path)
    
    # Define target variable
    target = 'treatment_outcome'
    
    # Feature selection — clinical features + encoded demographics
    # Note: Including demographics in baseline to demonstrate how the model
    # learns demographic-correlated patterns (which SHAP will reveal)
    feature_cols = [
        'age', 'systolic_bp', 'diastolic_bp', 'hba1c', 'ldl_cholesterol',
        'bmi', 'creatinine', 'heart_rate', 'smoking_status',
        'physical_activity_score', 'prior_hospitalizations',
        'medication_adherence'
    ]
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_insurance = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['race_encoded'] = le_race.fit_transform(df['race'])
    df['insurance_encoded'] = le_insurance.fit_transform(df['insurance_type'])
    
    feature_cols += ['gender_encoded', 'race_encoded', 'insurance_encoded']
    
    X = df[feature_cols].values
    y = df[target].values
    
    # Preserve demographics for subgroup analysis
    demographics = df[['patient_id', 'age', 'gender', 'race', 'insurance_type']].copy()
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demographics, test_size=0.25, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols, demo_test, df


def train_baseline_model(X_train, y_train):
    """
    Train a standard Random Forest classifier without fairness constraints.
    
    This deliberately uses default hyperparameters to demonstrate how
    off-the-shelf models can perpetuate diagnostic disparities.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
        
    Returns
    -------
    RandomForestClassifier
        Trained baseline model
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=None,  # No class balancing (intentional for baseline)
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_overall(model, X_test, y_test):
    """
    Compute overall model performance metrics.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
        
    Returns
    -------
    dict
        Performance metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_prob),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    return metrics, y_pred, y_prob


def evaluate_subgroups(model, X_test, y_test, demo_test):
    """
    Evaluate model performance across demographic subgroups to identify
    diagnostic disparities.
    
    Parameters
    ----------
    model : RandomForestClassifier
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    demo_test : pd.DataFrame
        Demographic information for test set
        
    Returns
    -------
    dict
        Subgroup-level performance metrics
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    subgroup_metrics = {}
    
    # Race subgroups
    for race in demo_test['race'].unique():
        mask = demo_test['race'].values == race
        if mask.sum() > 10:  # Minimum sample size
            subgroup_metrics[f'race_{race}'] = {
                'n': int(mask.sum()),
                'auc': float(roc_auc_score(y_test[mask], y_prob[mask])) if len(np.unique(y_test[mask])) > 1 else None,
                'accuracy': float(accuracy_score(y_test[mask], y_pred[mask])),
                'recall': float(recall_score(y_test[mask], y_pred[mask], zero_division=0))
            }
    
    # Age subgroups
    ages = demo_test['age'].values
    for label, mask in [('18-44', ages < 45), ('45-64', (ages >= 45) & (ages < 65)), ('65+', ages >= 65)]:
        if mask.sum() > 10:
            subgroup_metrics[f'age_{label}'] = {
                'n': int(mask.sum()),
                'auc': float(roc_auc_score(y_test[mask], y_prob[mask])) if len(np.unique(y_test[mask])) > 1 else None,
                'accuracy': float(accuracy_score(y_test[mask], y_pred[mask])),
                'recall': float(recall_score(y_test[mask], y_pred[mask], zero_division=0))
            }
    
    # Gender subgroups
    for gender in demo_test['gender'].unique():
        mask = demo_test['gender'].values == gender
        if mask.sum() > 10:
            subgroup_metrics[f'gender_{gender}'] = {
                'n': int(mask.sum()),
                'auc': float(roc_auc_score(y_test[mask], y_prob[mask])) if len(np.unique(y_test[mask])) > 1 else None,
                'accuracy': float(accuracy_score(y_test[mask], y_pred[mask])),
                'recall': float(recall_score(y_test[mask], y_pred[mask], zero_division=0))
            }
    
    return subgroup_metrics


def main():
    """Execute baseline model training and evaluation pipeline."""
    print("=" * 70)
    print("HEALTHCARE AI DIAGNOSTIC GAP — BASELINE MODEL")
    print("Random Forest Classifier (No Fairness Constraints)")
    print("=" * 70)
    print()
    
    # Load data
    print("[1/4] Loading and preprocessing synthetic clinical data...")
    X_train, X_test, y_train, y_test, feature_names, demo_test, full_df = load_and_preprocess()
    print(f"      Training set: {len(X_train)} patients")
    print(f"      Test set:     {len(X_test)} patients")
    print(f"      Features:     {len(feature_names)}")
    
    # Train model
    print("\n[2/4] Training baseline Random Forest classifier...")
    model = train_baseline_model(X_train, y_train)
    print(f"      Model: RandomForest (n_estimators=200, max_depth=12)")
    print(f"      Class weight: None (no balancing)")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"      5-Fold CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Overall evaluation
    print("\n[3/4] Evaluating overall model performance...")
    metrics, y_pred, y_prob = evaluate_overall(model, X_test, y_test)
    
    print(f"\n{'─' * 50}")
    print(f"  BASELINE MODEL PERFORMANCE")
    print(f"{'─' * 50}")
    print(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"{'─' * 50}")
    
    # Classification report
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Adverse Outcome', 'Positive Outcome'],
                                indent=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Adverse  Positive")
    print(f"    Actual Adverse  [{cm[0][0]:4d}    {cm[0][1]:4d}]")
    print(f"    Actual Positive [{cm[1][0]:4d}    {cm[1][1]:4d}]")
    
    # Subgroup analysis
    print(f"\n[4/4] Evaluating subgroup performance (diagnostic gap detection)...")
    subgroup_metrics = evaluate_subgroups(model, X_test, y_test, demo_test)
    
    print(f"\n{'─' * 50}")
    print(f"  SUBGROUP PERFORMANCE ANALYSIS")
    print(f"{'─' * 50}")
    
    print(f"\n  {'Subgroup':<20} {'N':>5} {'AUC':>7} {'Accuracy':>10} {'Recall':>8}")
    print(f"  {'─'*20} {'─'*5} {'─'*7} {'─'*10} {'─'*8}")
    
    for group, m in sorted(subgroup_metrics.items()):
        auc_str = f"{m['auc']:.4f}" if m['auc'] else "  N/A"
        print(f"  {group:<20} {m['n']:>5} {auc_str:>7} {m['accuracy']:>10.4f} {m['recall']:>8.4f}")
    
    # Identify disparities
    print(f"\n{'─' * 50}")
    print(f"  ⚠️  DIAGNOSTIC DISPARITIES DETECTED")
    print(f"{'─' * 50}")
    
    # Find max and min AUC across race subgroups
    race_aucs = {k: v['auc'] for k, v in subgroup_metrics.items() 
                 if k.startswith('race_') and v['auc'] is not None}
    if race_aucs:
        max_race = max(race_aucs, key=race_aucs.get)
        min_race = min(race_aucs, key=race_aucs.get)
        gap = race_aucs[max_race] - race_aucs[min_race]
        print(f"\n  Race AUC Gap: {gap:.4f}")
        print(f"    Highest: {max_race} (AUC={race_aucs[max_race]:.4f})")
        print(f"    Lowest:  {min_race} (AUC={race_aucs[min_race]:.4f})")
    
    age_aucs = {k: v['auc'] for k, v in subgroup_metrics.items() 
                if k.startswith('age_') and v['auc'] is not None}
    if age_aucs:
        max_age = max(age_aucs, key=age_aucs.get)
        min_age = min(age_aucs, key=age_aucs.get)
        gap = age_aucs[max_age] - age_aucs[min_age]
        print(f"\n  Age AUC Gap: {gap:.4f}")
        print(f"    Highest: {max_age} (AUC={age_aucs[max_age]:.4f})")
        print(f"    Lowest:  {min_age} (AUC={age_aucs[min_age]:.4f})")
    
    # Save model and metrics
    output_dir = Path(__file__).parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "baseline_model.joblib"
    joblib.dump(model, model_path)
    
    metrics_output = {
        'overall': metrics,
        'subgroups': subgroup_metrics,
        'feature_names': feature_names,
        'cv_scores': cv_scores.tolist()
    }
    
    metrics_path = output_dir / "baseline_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"\n  Model saved to: {model_path}")
    print(f"  Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("✓ Baseline model training complete.")
    print("  Overall AUC: {:.4f} — Significant subgroup disparities detected.".format(metrics['auc_roc']))
    print("  Proceed to SHAP analysis for interpretability insights.")
    print("=" * 70)
    
    return model, metrics, subgroup_metrics


if __name__ == "__main__":
    model, metrics, subgroup_metrics = main()
