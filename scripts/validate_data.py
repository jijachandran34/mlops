"""
Data validation checks for CI pipeline
"""
import pandas as pd
import sys
import json
import os

def validate_data(file_path):
    """
    Perform data validation checks
    Returns True if all checks pass, False otherwise
    """
    results = {
        "file": file_path,
        "checks_passed": [],
        "checks_failed": [],
        "warnings": []
    }
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Check 1: Required columns
        required_cols = ['target']
        if all(col in df.columns for col in required_cols):
            results["checks_passed"].append("Required columns present")
        else:
            results["checks_failed"].append("Missing required columns")
        
        # Check 2: No null values
        if df.isnull().sum().sum() == 0:
            results["checks_passed"].append("No null values")
        else:
            null_count = df.isnull().sum().sum()
            results["checks_failed"].append(f"Found {null_count} null values")
        
        # Check 3: Expected number of classes
        if 'target' in df.columns:
            n_classes = df['target'].nunique()
            if n_classes == 3:
                results["checks_passed"].append("Correct number of classes (3)")
            else:
                results["warnings"].append(f"Expected 3 classes, found {n_classes}")
        
        # Check 4: Data shape
        if len(df) > 0:
            results["checks_passed"].append(f"Data shape: {df.shape}")
        else:
            results["checks_failed"].append("Empty dataframe")
        
        # Check 5: Feature ranges (basic sanity)
        feature_cols = [col for col in df.columns if col != 'target']
        for col in feature_cols:
            if df[col].std() == 0:
                results["warnings"].append(f"Zero variance in {col}")
        
        # Summary
        all_passed = len(results["checks_failed"]) == 0
        results["validation_passed"] = all_passed
        results["summary"] = {
            "passed": len(results["checks_passed"]),
            "failed": len(results["checks_failed"]),
            "warnings": len(results["warnings"])
        }
        
        # Save results
        os.makedirs("reports", exist_ok=True)
        with open("reports/validation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print results
        print("\n=== Data Validation Results ===")
        print(f"File: {file_path}")
        print(f"\n✓ Passed checks ({len(results['checks_passed'])}):")
        for check in results["checks_passed"]:
            print(f"  - {check}")
        
        if results["checks_failed"]:
            print(f"\n✗ Failed checks ({len(results['checks_failed'])}):")
            for check in results["checks_failed"]:
                print(f"  - {check}")
        
        if results["warnings"]:
            print(f"\n⚠ Warnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"  - {warning}")
        
        print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed
        
    except Exception as e:
        print(f"Error during validation: {e}")
        results["checks_failed"].append(f"Exception: {str(e)}")
        results["validation_passed"] = False
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <data_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    passed = validate_data(file_path)
    sys.exit(0 if passed else 1)
