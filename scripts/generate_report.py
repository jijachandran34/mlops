"""
Generate markdown report for CML
"""
import json
import os
import glob

def generate_report():
    """Generate comprehensive report from all metrics"""
    
    report = []
    report.append("# IRIS MLOps Pipeline Report\n")
    
    # Validation results
    if os.path.exists("reports/validation_results.json"):
        with open("reports/validation_results.json") as f:
            validation = json.load(f)
        
        report.append("## Data Validation\n")
        report.append(f"**Status**: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}\n")
        report.append(f"- Checks Passed: {validation['summary']['passed']}")
        report.append(f"- Checks Failed: {validation['summary']['failed']}")
        report.append(f"- Warnings: {validation['summary']['warnings']}\n")
    
    # Training metrics
    report.append("## Model Performance by Poison Level\n")
    report.append("| Poison Level | Accuracy | F1 Score | Precision | Recall |")
    report.append("|--------------|----------|----------|-----------|--------|")
    
    # Find all metric files
    metric_files = glob.glob("reports/metrics_poison_*.json")
    metric_files.sort()
    
    metrics_data = []
    for mfile in metric_files:
        with open(mfile) as f:
            metrics = json.load(f)
            metrics_data.append(metrics)
            report.append(
                f"| {metrics['poison_level']}% | "
                f"{metrics['accuracy']:.4f} | "
                f"{metrics['f1_score']:.4f} | "
                f"{metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} |"
            )
    
    # Analysis
    if len(metrics_data) > 1:
        report.append("\n## Impact Analysis\n")
        clean_metrics = next((m for m in metrics_data if m['poison_level'] == 0), None)
        if clean_metrics:
            report.append("**Accuracy Degradation:**\n")
            for m in metrics_data:
                if m['poison_level'] > 0:
                    degradation = (clean_metrics['accuracy'] - m['accuracy']) * 100
                    report.append(
                        f"- At {m['poison_level']}% poison: "
                        f"{degradation:.2f}% accuracy drop"
                    )
    
    # Save report
    report_text = "\n".join(report)
    with open("reports/report.md", "w") as f:
        f.write(report_text)
    
    print(report_text)
    return report_text

if __name__ == "__main__":
    generate_report()
