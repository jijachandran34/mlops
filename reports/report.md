# IRIS MLOps Pipeline Report

## Model Performance by Poison Level

| Poison Level | Accuracy | F1 Score | Precision | Recall |
|--------------|----------|----------|-----------|--------|
| 0.0% | 0.9048 | 0.8942 | 0.9238 | 0.9048 |
| 0.0% | 0.9048 | 0.8942 | 0.9238 | 0.9048 |
| 10.0% | 0.8095 | 0.8067 | 0.8730 | 0.8095 |
| 5.0% | 0.8571 | 0.8511 | 0.8961 | 0.8571 |
| 50.0% | 0.5714 | 0.5770 | 0.5866 | 0.5714 |

## Impact Analysis

**Accuracy Degradation:**

- At 10.0% poison: 9.52% accuracy drop
- At 5.0% poison: 4.76% accuracy drop
- At 50.0% poison: 33.33% accuracy drop