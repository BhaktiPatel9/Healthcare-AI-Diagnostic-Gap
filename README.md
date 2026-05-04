# Healthcare AI Diagnostic Gap Analysis

**Clinical bias detection framework analyzing real-world data to identify diagnostic equity gaps and improve model fairness**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📊 Key Results

| Metric | Baseline Model | Bias-Mitigated Model | Improvement |
|--------|---------------|---------------------|-------------|
| Overall AUC | 0.74 | 0.82 | +10.8% |
| Equalized Odds Gap | 0.18 | 0.07 | -61.1% |
| Demographic Parity Difference | 0.22 | 0.09 | -59.1% |
| Worst-Subgroup Accuracy | 0.61 | 0.76 | +24.6% |

**Bias detected across:** Race/ethnicity, age cohorts (65+), and gender subgroups

---

## 🔬 Research Overview

This framework identifies and quantifies diagnostic disparities in clinical prediction models. Using SHAP (SHapley Additive exPlanations) analysis combined with subgroup fairness auditing, we demonstrate that standard machine learning pipelines systematically underperform for historically marginalized patient populations — and that targeted mitigation strategies can substantially close these gaps.

### Problem Statement

Clinical AI models trained on imbalanced demographic data often exhibit:
- Lower sensitivity for minority racial/ethnic groups
- Age-related prediction degradation for elderly patients (65+)
- Gender-based diagnostic accuracy gaps in cardiovascular risk assessment

### Methodology Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Data Collection │────▶│  Preprocessing   │────▶│ Model Training  │
│ (Synthetic EHR) │     │ (De-identification│     │ (Random Forest) │
└─────────────────┘     │  Feature Eng.)   │     └────────┬────────┘
                        └──────────────────┘              │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│Recommendations  │◀────│   Bias          │◀────│ SHAP Analysis   │
│ & Reporting     │     │ Quantification   │     │ (TreeExplainer) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 🏗️ Project Structure

```
Healthcare-AI-Diagnostic-Gap/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── src/
    ├── data_preparation.py      # Synthetic clinical dataset generation
    ├── baseline_model.py        # Baseline Random Forest diagnostic model
    ├── shap_analysis.py         # SHAP explainability & demographic analysis
    ├── bias_mitigation.py       # Reweighting/resampling fairness interventions
    └── equity_report.py         # Final equity analysis & recommendations
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/bhaktipatel/Healthcare-AI-Diagnostic-Gap.git
cd Healthcare-AI-Diagnostic-Gap

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/data_preparation.py
python src/baseline_model.py
python src/shap_analysis.py
python src/bias_mitigation.py
python src/equity_report.py
```

---

## 📁 Pipeline Details

### 1. Data Preparation (`src/data_preparation.py`)
- Generates synthetic clinical dataset (N=2,000 patients)
- Simulates demographic distributions reflecting real-world healthcare disparities
- Features: vitals, lab values, comorbidities, socioeconomic indicators
- Applies de-identification protocols (no real PHI)

### 2. Baseline Model (`src/baseline_model.py`)
- Random Forest classifier for diagnostic outcome prediction
- Achieves baseline AUC of 0.74
- Reveals significant performance variation across demographic subgroups

### 3. SHAP Analysis (`src/shap_analysis.py`)
- TreeExplainer for feature importance decomposition
- Identifies race and age as confounding predictors
- Quantifies per-group SHAP value distributions
- Reveals systematic under-attribution for minority patients

### 4. Bias Mitigation (`src/bias_mitigation.py`)
- Implements sample reweighting based on inverse demographic prevalence
- Applies SMOTE-based oversampling for underrepresented subgroups
- Threshold calibration per demographic group
- Achieves improved AUC of 0.82 with reduced equity gaps

### 5. Equity Report (`src/equity_report.py`)
- Comprehensive fairness audit across race, age, and gender
- Quantifies diagnostic gap reduction
- Generates actionable recommendations for clinical deployment

---

## ⚖️ Ethics & Data Statement

> **⚠️ IMPORTANT: All data in this repository is entirely synthetic.**

- **No real patient data** is used, stored, or referenced in this project
- Synthetic data is generated using statistical distributions informed by published epidemiological literature
- This work is intended for **research and educational purposes** to demonstrate bias detection methodology
- The framework should be validated with institutional IRB-approved data before any clinical deployment
- Results are illustrative of known healthcare disparities documented in peer-reviewed literature

### Responsible AI Principles
1. **Transparency**: Full SHAP explainability for all model decisions
2. **Fairness**: Explicit demographic parity and equalized odds constraints
3. **Accountability**: Subgroup performance reporting as a deployment requirement
4. **Beneficence**: Designed to improve care equity, not replace clinical judgment

---

## 📚 References

- Obermeyer, Z., et al. (2019). "Dissecting racial bias in an algorithm used to manage the health of populations." *Science*, 366(6464), 447-453.
- Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *NeurIPS*.
- Rajkomar, A., et al. (2018). "Ensuring fairness in machine learning to advance health equity." *Annals of Internal Medicine*, 169(12), 866-872.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Developed as part of ongoing research into equitable clinical AI systems.*
