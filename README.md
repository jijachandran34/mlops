# IRIS MLOps Pipeline with Data Poisoning Analysis

## Overview
This project demonstrates an MLOps pipeline for the IRIS dataset with:
- Data versioning using DVC
- CI/CD using GitHub Actions
- Automated reporting with CML
- Data poisoning experiments at various levels (5%, 10%, 50%)
- MLFlow experiment tracking

## Project Structure
```
.
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI pipeline
├── data/                      # Data files (tracked by DVC)
├── scripts/
│   ├── fetch_data.py         # Fetch data from GCS
│   ├── poison_data.py        # Create poisoned datasets
│   ├── train.py              # Train models and log metrics
│   ├── validate_data.py      # Data validation checks
│   └── generate_report.py    # Generate markdown reports
├── reports/                   # Generated reports and metrics
├── dvc.yaml                  # DVC pipeline definition
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup Instructions

### 1. Clone and Setup
```bash
git clone https://github.com/jijachandran34/mlops.git
cd mlops
git checkout week8
pip install -r requirements.txt
```

### 2. Initialize DVC
```bash
dvc init
dvc remote add -d myremote gs://ml_ops_wk1_826/dvc-storage
```

### 3. Run Pipeline Locally
```bash
# Fetch data
python3 scripts/fetch_data.py

# Run complete DVC pipeline
dvc repro

# Generate report
python3 scripts/generate_report.py
```

### 4. View MLFlow Results
```bash
mlflow ui
# Open browser to http://localhost:5000
```

## Data Poisoning Experiments

### Poisoning Strategy
Random feature replacement: Randomly selected samples have their features replaced with random values within the feature range.

### Poison Levels Tested
- **0%**: Clean baseline data
- **5%**: Mild poisoning
- **10%**: Moderate poisoning
- **50%**: Severe poisoning

## Mitigation Strategies for Data Poisoning

### 1. Detection Methods
- **Statistical Outlier Detection**: Use z-scores, IQR, or isolation forests to identify anomalous samples
- **Data Validation Checks**: Implement range checks, distribution comparisons, and schema validation
- **Model-Based Detection**: Train clean models and flag samples with high prediction uncertainty

### 2. Prevention Approaches
- **Data Provenance Tracking**: Maintain audit logs of data sources and transformations
- **Multi-Source Validation**: Cross-validate data from multiple independent sources
- **Input Sanitization**: Apply strict input validation at data ingestion points

### 3. Robust Training
- **Outlier-Resistant Loss Functions**: Use Huber loss or trimmed loss functions
- **Data Sanitization**: Remove detected outliers before training
- **Ensemble Methods**: Use multiple models to reduce impact of poisoned samples

### 4. Continuous Monitoring
- **Drift Detection**: Monitor for data distribution changes over time
- **Performance Monitoring**: Alert on unexpected accuracy drops
- **Regular Audits**: Periodic manual review of data quality

## Data Quantity vs Quality Trade-offs

### Impact of Quality Degradation
As data quality decreases, more data is required to maintain model performance:

```
Quality Level | Data Multiplier | Example
--------------|-----------------|------------------
100% (Clean)  | 1x             | 1000 samples
95% (5% bad)  | 1.2-1.5x       | 1200-1500 samples
90% (10% bad) | 1.5-2x         | 1500-2000 samples
50% (50% bad) | 3-5x           | 3000-5000 samples
```

### Key Insights
1. **Non-linear relationship**: Quality loss requires exponentially more data
2. **Diminishing returns**: After a threshold, more data can't compensate for poor quality
3. **Cost considerations**: Focus on quality over quantity for cost-effective ML

## CI/CD Pipeline

The GitHub Actions pipeline automatically:
1. Validates data quality
2. Runs the complete DVC pipeline
3. Trains models at all poisoning levels
4. Generates comparison reports
5. Posts results as PR comments using CML

## Results Interpretation

### Expected Outcomes
- **5% poisoning**: 2-5% accuracy drop
- **10% poisoning**: 5-10% accuracy drop
- **50% poisoning**: 20-40% accuracy drop

### Key Metrics
- **Accuracy**: Overall correctness
- **F1 Score**: Balance of precision and recall
- **Precision**: False positive rate
- **Recall**: False negative rate

## References
- [DVC Documentation](https://dvc.org/doc)
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [CML Documentation](https://cml.dev/)
