"""
Healthcare AI Diagnostic Gap Analysis
======================================
Module: Data Preparation
Purpose: Generate synthetic clinical dataset simulating real-world demographic
         distributions and diagnostic disparities in healthcare outcomes.

All data is entirely synthetic — no real patient health information (PHI) is
used or referenced. Distributions are informed by published epidemiological
literature to reflect documented healthcare disparities.

Author: Bhakti Patel
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_PATIENTS = 2000  # Total synthetic patient cohort size

# Demographic distributions (informed by US Census & CMS data)
RACE_DISTRIBUTION = {
    'White': 0.58,
    'Black': 0.18,
    'Hispanic': 0.14,
    'Asian': 0.07,
    'Other': 0.03
}

GENDER_DISTRIBUTION = {'Female': 0.52, 'Male': 0.48}

AGE_PARAMS = {'mean': 55, 'std': 18, 'min': 18, 'max': 95}

# Diagnostic categories (cardiovascular risk assessment)
DIAGNOSES = [
    'Hypertension', 'Type 2 Diabetes', 'Coronary Artery Disease',
    'Heart Failure', 'Atrial Fibrillation', 'Healthy Control'
]


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_demographics(n: int) -> pd.DataFrame:
    """
    Generate synthetic patient demographics with realistic distributions.
    
    Parameters
    ----------
    n : int
        Number of synthetic patients to generate
        
    Returns
    -------
    pd.DataFrame
        DataFrame with patient_id, age, gender, race columns
    """
    # Patient IDs (de-identified format)
    patient_ids = [f"SYN-{i:05d}" for i in range(1, n + 1)]
    
    # Age distribution (truncated normal)
    ages = np.random.normal(AGE_PARAMS['mean'], AGE_PARAMS['std'], n)
    ages = np.clip(ages, AGE_PARAMS['min'], AGE_PARAMS['max']).astype(int)
    
    # Gender assignment
    genders = np.random.choice(
        list(GENDER_DISTRIBUTION.keys()),
        size=n,
        p=list(GENDER_DISTRIBUTION.values())
    )
    
    # Race/ethnicity assignment
    races = np.random.choice(
        list(RACE_DISTRIBUTION.keys()),
        size=n,
        p=list(RACE_DISTRIBUTION.values())
    )
    
    return pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        'race': races
    })


def generate_clinical_features(demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic clinical measurements with demographic-correlated
    distributions that simulate real-world diagnostic disparities.
    
    This intentionally introduces systematic differences to model known
    healthcare access and outcome disparities documented in literature.
    
    Parameters
    ----------
    demographics : pd.DataFrame
        Patient demographics from generate_demographics()
        
    Returns
    -------
    pd.DataFrame
        Clinical features including lab values, vitals, and risk scores
    """
    n = len(demographics)
    
    # Base clinical measurements
    systolic_bp = np.random.normal(130, 20, n)
    diastolic_bp = np.random.normal(80, 12, n)
    hba1c = np.random.normal(6.2, 1.5, n)
    ldl_cholesterol = np.random.normal(120, 35, n)
    bmi = np.random.normal(28, 6, n)
    creatinine = np.random.normal(1.0, 0.3, n)
    
    # Introduce age-correlated clinical worsening
    age_factor = (demographics['age'].values - 40) / 50
    systolic_bp += age_factor * 15
    hba1c += np.maximum(age_factor, 0) * 0.8
    ldl_cholesterol += age_factor * 20
    
    # Simulate documented disparities in clinical measurements
    # (Based on published literature on healthcare outcome gaps)
    for idx, row in demographics.iterrows():
        # Higher baseline cardiovascular risk in Black patients (documented disparity)
        if row['race'] == 'Black':
            systolic_bp[idx] += np.random.normal(8, 3)
            creatinine[idx] += np.random.normal(0.1, 0.05)
        
        # Hispanic patients: higher diabetes prevalence (CDC documented)
        elif row['race'] == 'Hispanic':
            hba1c[idx] += np.random.normal(0.4, 0.2)
            bmi[idx] += np.random.normal(1.5, 1.0)
        
        # Gender-based cardiovascular presentation differences
        if row['gender'] == 'Female' and row['age'] < 55:
            ldl_cholesterol[idx] -= np.random.normal(10, 5)
    
    # Composite lab values panel
    clinical_features = pd.DataFrame({
        'systolic_bp': np.clip(systolic_bp, 90, 200).round(1),
        'diastolic_bp': np.clip(diastolic_bp, 50, 120).round(1),
        'hba1c': np.clip(hba1c, 4.0, 14.0).round(2),
        'ldl_cholesterol': np.clip(ldl_cholesterol, 40, 250).round(1),
        'bmi': np.clip(bmi, 16, 55).round(1),
        'creatinine': np.clip(creatinine, 0.4, 4.0).round(2),
        'heart_rate': np.clip(np.random.normal(75, 12, n), 45, 130).round(0).astype(int),
        'smoking_status': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'physical_activity_score': np.clip(np.random.normal(5, 2, n), 0, 10).round(1),
        'insurance_type': np.random.choice(
            ['Private', 'Medicare', 'Medicaid', 'Uninsured'],
            n, p=[0.45, 0.28, 0.18, 0.09]
        ),
        'prior_hospitalizations': np.random.poisson(1.2, n),
        'medication_adherence': np.clip(np.random.beta(7, 3, n), 0, 1).round(3)
    })
    
    return clinical_features


def generate_outcomes(demographics: pd.DataFrame, 
                      clinical: pd.DataFrame) -> pd.DataFrame:
    """
    Generate treatment outcomes and readmission risk with systematic
    disparities that reflect documented healthcare inequities.
    
    The outcome generation intentionally creates diagnostic gaps that
    the bias detection framework will identify and quantify.
    
    Parameters
    ----------
    demographics : pd.DataFrame
        Patient demographics
    clinical : pd.DataFrame
        Clinical features
        
    Returns
    -------
    pd.DataFrame
        Outcomes including diagnosis, treatment_outcome, readmission_risk
    """
    n = len(demographics)
    
    # Generate diagnosis based on clinical features
    diagnoses = []
    for idx in range(n):
        risk_score = 0
        risk_score += (clinical.iloc[idx]['systolic_bp'] - 120) / 30
        risk_score += (clinical.iloc[idx]['hba1c'] - 5.7) / 2
        risk_score += (clinical.iloc[idx]['ldl_cholesterol'] - 100) / 50
        risk_score += (clinical.iloc[idx]['bmi'] - 25) / 10
        risk_score += clinical.iloc[idx]['smoking_status'] * 1.5
        risk_score += (demographics.iloc[idx]['age'] - 40) / 30
        
        # Assign diagnosis based on composite risk
        if risk_score > 4.5:
            diagnoses.append(np.random.choice(
                ['Coronary Artery Disease', 'Heart Failure'], p=[0.6, 0.4]))
        elif risk_score > 3.0:
            diagnoses.append(np.random.choice(
                ['Hypertension', 'Atrial Fibrillation'], p=[0.7, 0.3]))
        elif risk_score > 1.5:
            diagnoses.append(np.random.choice(
                ['Hypertension', 'Type 2 Diabetes'], p=[0.5, 0.5]))
        else:
            diagnoses.append('Healthy Control')
    
    # Treatment outcome (1 = positive, 0 = adverse)
    # Introduce systematic disparity: lower positive outcomes for certain groups
    treatment_outcomes = np.zeros(n)
    for idx in range(n):
        base_prob = 0.72  # Baseline positive outcome probability
        
        # Clinical severity adjustment
        base_prob -= (clinical.iloc[idx]['hba1c'] - 5.7) * 0.03
        base_prob -= clinical.iloc[idx]['prior_hospitalizations'] * 0.04
        base_prob += clinical.iloc[idx]['medication_adherence'] * 0.15
        
        # Documented disparity: lower treatment success rates
        # (reflects access, implicit bias, social determinants)
        if demographics.iloc[idx]['race'] == 'Black':
            base_prob -= 0.08
        elif demographics.iloc[idx]['race'] == 'Hispanic':
            base_prob -= 0.05
        
        # Age-related treatment complexity
        if demographics.iloc[idx]['age'] > 65:
            base_prob -= 0.06
        
        # Insurance access effects
        if clinical.iloc[idx]['insurance_type'] == 'Uninsured':
            base_prob -= 0.12
        elif clinical.iloc[idx]['insurance_type'] == 'Medicaid':
            base_prob -= 0.05
        
        treatment_outcomes[idx] = np.random.binomial(1, np.clip(base_prob, 0.1, 0.95))
    
    # Readmission risk (continuous 0-1)
    readmission_base = np.random.beta(2, 5, n)
    readmission_risk = readmission_base + (1 - treatment_outcomes) * 0.2
    readmission_risk = np.clip(readmission_risk, 0, 1).round(3)
    
    return pd.DataFrame({
        'diagnosis': diagnoses,
        'treatment_outcome': treatment_outcomes.astype(int),
        'readmission_risk': readmission_risk
    })


def apply_deidentification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply de-identification protocols to ensure no PHI leakage.
    
    Even though data is synthetic, this demonstrates proper de-identification
    methodology consistent with HIPAA Safe Harbor standards.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined patient dataset
        
    Returns
    -------
    pd.DataFrame
        De-identified dataset
    """
    # Generalize age for patients 90+ (HIPAA Safe Harbor)
    df.loc[df['age'] > 89, 'age'] = 90
    
    # Ensure no direct identifiers exist
    assert 'name' not in df.columns
    assert 'ssn' not in df.columns
    assert 'address' not in df.columns
    assert 'phone' not in df.columns
    
    # Add de-identification flag
    df['deidentified'] = True
    df['data_source'] = 'synthetic'
    
    return df


def main():
    """Execute the full data preparation pipeline."""
    print("=" * 70)
    print("HEALTHCARE AI DIAGNOSTIC GAP — DATA PREPARATION")
    print("Synthetic Clinical Dataset Generation")
    print("=" * 70)
    print()
    
    # Step 1: Generate demographics
    print("[1/5] Generating patient demographics...")
    demographics = generate_demographics(N_PATIENTS)
    print(f"      Generated {len(demographics)} synthetic patient records")
    
    # Step 2: Generate clinical features
    print("[2/5] Generating clinical measurements...")
    clinical = generate_clinical_features(demographics)
    print(f"      {len(clinical.columns)} clinical features created")
    
    # Step 3: Generate outcomes
    print("[3/5] Generating treatment outcomes with disparity simulation...")
    outcomes = generate_outcomes(demographics, clinical)
    
    # Step 4: Combine dataset
    print("[4/5] Assembling complete dataset...")
    dataset = pd.concat([demographics, clinical, outcomes], axis=1)
    
    # Step 5: De-identification
    print("[5/5] Applying de-identification protocols...")
    dataset = apply_deidentification(dataset)
    
    # Save dataset
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "synthetic_clinical_data.csv"
    dataset.to_csv(output_path, index=False)
    print(f"\n      Dataset saved to: {output_path}")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "=" * 70)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 Total Patients: {len(dataset)}")
    print(f"📊 Features: {len(dataset.columns)}")
    print(f"📊 Data Source: 100% Synthetic (No Real PHI)")
    
    print("\n--- Demographic Distribution ---")
    print(f"\nRace/Ethnicity:")
    race_counts = dataset['race'].value_counts()
    for race, count in race_counts.items():
        print(f"  {race:12s}: {count:4d} ({count/len(dataset)*100:.1f}%)")
    
    print(f"\nGender:")
    gender_counts = dataset['gender'].value_counts()
    for gender, count in gender_counts.items():
        print(f"  {gender:12s}: {count:4d} ({count/len(dataset)*100:.1f}%)")
    
    print(f"\nAge Distribution:")
    print(f"  Mean: {dataset['age'].mean():.1f} years")
    print(f"  Std:  {dataset['age'].std():.1f} years")
    print(f"  Range: {dataset['age'].min()} - {dataset['age'].max()} years")
    
    print("\n--- Clinical Summary ---")
    print(f"\nDiagnosis Distribution:")
    dx_counts = dataset['diagnosis'].value_counts()
    for dx, count in dx_counts.items():
        print(f"  {dx:28s}: {count:4d} ({count/len(dataset)*100:.1f}%)")
    
    print(f"\nTreatment Outcomes:")
    pos_rate = dataset['treatment_outcome'].mean()
    print(f"  Positive outcome rate: {pos_rate:.3f}")
    print(f"  Adverse outcome rate:  {1 - pos_rate:.3f}")
    
    print("\n--- Disparity Indicators ---")
    print(f"\nTreatment Outcome by Race:")
    for race in RACE_DISTRIBUTION.keys():
        subset = dataset[dataset['race'] == race]
        rate = subset['treatment_outcome'].mean()
        print(f"  {race:12s}: {rate:.3f} (n={len(subset)})")
    
    print(f"\nTreatment Outcome by Age Group:")
    for label, mask in [('18-44', dataset['age'] < 45),
                        ('45-64', (dataset['age'] >= 45) & (dataset['age'] < 65)),
                        ('65+', dataset['age'] >= 65)]:
        subset = dataset[mask]
        rate = subset['treatment_outcome'].mean()
        print(f"  {label:12s}: {rate:.3f} (n={len(subset)})")
    
    print(f"\nReadmission Risk (mean by race):")
    for race in RACE_DISTRIBUTION.keys():
        subset = dataset[dataset['race'] == race]
        risk = subset['readmission_risk'].mean()
        print(f"  {race:12s}: {risk:.3f}")
    
    print("\n" + "=" * 70)
    print("✓ Data preparation complete. Ready for baseline model training.")
    print("=" * 70)
    
    return dataset


if __name__ == "__main__":
    dataset = main()
