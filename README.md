# Healthcare AI Diagnostics: Accuracy & Bias Evaluation

**Research Impact:** Quantified a 10% improvement in diagnostic accuracy using AI models while identifying critical algorithmic bias risks in demographic subgroups.

## Project Overview
As AI tools are increasingly adopted in clinical settings, establishing a baseline of trust and accuracy is critical. This research project evaluated the performance of an AI diagnostic classification model against traditional medical baselines using a dataset of 200+ clinical records. The investigation focused not only on overall accuracy but also on False Negative rates and data privacy/compliance risks (HIPAA frameworks).

## Tech Stack & Methodology
* **Language:** Python (Pandas, NumPy)
* **Machine Learning Evaluation:** Scikit-learn (Confusion Matrices, Classification Reports)
* **Statistical Focus:** Algorithmic fairness, demographic bias detection, and process optimization.

## Key Findings
1. **The Accuracy Gap:** The AI model outperformed the traditional baseline, jumping from an accuracy score of **0.74 to 0.82**.
2. **The Bias Risk:** While overall accuracy improved, subgroup analysis revealed that the model had a higher false-negative rate for specific minority demographics, highlighting a need for more diverse training data before production deployment.
3. **Compliance Context:** Evaluated the data pipeline to ensure patient PII/PHI was anonymized prior to model ingestion.

---
*Note: The dataset and exact medical parameters used in this repository have been fully anonymized and mocked to comply with healthcare data privacy standards.*
