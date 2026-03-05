"""
=============================================================================
Script: model_evaluation.py
Purpose: Evaluate AI diagnostic predictions against traditional baselines.
Focus: Accuracy improvement, False Negative rates, and subgroup bias detection.
=============================================================================
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_diagnostic_model(df):
    print("--- 🏥 Clinical AI Evaluation Report ---")
    
    # 1. Overall Accuracy Comparison
    baseline_acc = accuracy_score(df['true_diagnosis'], df['traditional_baseline'])
    ai_acc = accuracy_score(df['true_diagnosis'], df['ai_prediction'])
    
    print(f"\n[1] OVERALL ACCURACY")
    print(f"Traditional Method: {baseline_acc:.2f}")
    print(f"AI Model Method:    {ai_acc:.2f}")
    print(f"Improvement Gap:    +{(ai_acc - baseline_acc) * 100:.1f}%")
    
    # 2. Risk Analysis: False Negatives (Critical in Healthcare)
    # confusion_matrix returns: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(df['true_diagnosis'], df['ai_prediction']).ravel()
    
    print(f"\n[2] CLINICAL RISK ASSESSMENT (AI Model)")
    print(f"False Negatives (Missed Diagnoses): {fn}")
    print(f"False Positives (False Alarms):     {fp}")
    
    if fn > (len(df) * 0.05):
        print("⚠️ Warning: False Negative rate exceeds 5% acceptable clinical threshold.")
    
    # 3. Bias & Fairness Check (Evaluating a specific demographic subgroup)
    # Checking if the model performs worse on 'Subgroup B'
    subgroup_b = df[df['demographic_group'] == 'Group_B']
    sub_ai_acc = accuracy_score(subgroup_b['true_diagnosis'], subgroup_b['ai_prediction'])
    
    print(f"\n[3] ALGORITHMIC BIAS CHECK")
    print(f"Accuracy for Subgroup B: {sub_ai_acc:.2f}")
    
    if sub_ai_acc < (ai_acc - 0.05):
        print("🚨 BIAS DETECTED: Model underperforms on Subgroup B by more than 5%. Recalibration required.")
    else:
        print("✅ Fairness Check Passed: Consistent performance across demographics.")

# Simulating the 200+ clinical records dataset
if __name__ == "__main__":
    # Mocking a small sample for execution
    data = {
        'patient_id': range(1, 11),
        'demographic_group': ['Group_A', 'Group_B', 'Group_A', 'Group_B', 'Group_A', 
                              'Group_B', 'Group_A', 'Group_B', 'Group_A', 'Group_B'],
        'true_diagnosis':       [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        'traditional_baseline': [0, 0, 1, 0, 0, 1, 1, 0, 1, 0], # Accuracy ~ 0.60
        'ai_prediction':        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  # Accuracy ~ 0.80
    }
    
    clinical_df = pd.DataFrame(data)
    evaluate_diagnostic_model(clinical_df)
