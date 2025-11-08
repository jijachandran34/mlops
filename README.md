MLOps Pipeline — Iris Classification

This repository implements an end-to-end MLOps workflow for an Iris classification model, including data versioning (DVC), model evaluation, and Continuous Integration (CI) using GitHub Actions and Google Cloud Storage (GCS).

mlops/
  - src/
  - data_validation.py        # Data schema & integrity validation
  - evaluate.py               # Model evaluation logic (metrics, accuracy, etc.)
  -   tests/
  -   test_data_validation.py   # Unit tests for data validation
  -   test_evaluation.py        # Unit tests for model evaluation
  -   .github/workflows/
  -   ci.yml                # CI workflow (GitHub Actions)
  -   modify_iris_mean.py           # Script to modify dataset versions
  -   requirements.txt              # Python dependencies
  -   iris.csv.dvc                  # DVC pointer to dataset stored in GCS
  -   dvc/config                   # DVC configuration (remote: GCS bucket)

**Setup of CI pipeline**

1. Clone the Repository
2. Create and Activate a Virtual Environment
3. Install Dependencies
4. Authenticate with Google Cloud
5. Verify Google Cloud Authentication
6. Connect to DVC Remote (Google Cloud Storage)
7. Pull Data and Model from DVC Remote
8. Tests Locally
