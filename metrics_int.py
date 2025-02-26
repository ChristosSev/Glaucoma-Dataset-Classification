import json


def analyze_metrics(metrics_file, output_file="model_analysis.txt"):
    """Reads model metrics from a JSON file and writes an analysis report to a text file."""

    # Load metrics from the file
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Extract overall metrics
    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    f1_score = metrics.get("f1_score", 0)

    # Extract per-class metrics
    per_class_metrics = metrics.get("per_class", {})

    # Start report
    report = []
    report.append("AI Model Performance Analysis\n")
    report.append("=" * 50 + "\n")

    # Overall model metrics
    report.append(f"Overall Accuracy: {accuracy:.4f}\n")
    report.append(f"Precision: {precision:.4f}\n")
    report.append(f"Recall: {recall:.4f}\n")
    report.append(f"F1 Score: {f1_score:.4f}\n\n")

    # Interpretation of results
    if accuracy > 0.90:
        report.append("The model demonstrates high accuracy.\n")
    elif accuracy > 0.80:
        report.append("The model performs well but may require further improvements.\n")
    else:
        report.append("The model has low accuracy and requires optimization.\n")

    if precision < recall:
        report.append(
            "The model prioritizes recall over precision, indicating a tendency to classify more positive instances, even at the cost of false positives.\n")
    elif precision > recall:
        report.append(
            "The model prioritizes precision, minimizing false positives but potentially missing some true positives.\n")

    if f1_score < 0.75:
        report.append(
            "The balance between precision and recall is not optimal. Further adjustments to training data or model architecture may be necessary.\n")
    else:
        report.append("The model achieves a good balance between precision and recall.\n")

    # Per-class performance
    report.append("\nPer-Class Performance Analysis:\n")
    report.append("=" * 50 + "\n")

    worst_f1_class = None
    lowest_f1 = 1.0  # Initialize to a high value for comparison

    for class_name, scores in per_class_metrics.items():
        class_precision = scores.get("precision", 0)
        class_recall = scores.get("recall", 0)
        class_f1 = scores.get("f1_score", 0)

        report.append(f"Class: {class_name}\n")
        report.append(f"  - Precision: {class_precision:.4f}\n")
        report.append(f"  - Recall: {class_recall:.4f}\n")
        report.append(f"  - F1 Score: {class_f1:.4f}\n\n")

        if class_f1 < lowest_f1:
            lowest_f1 = class_f1
            worst_f1_class = class_name

    if worst_f1_class:
        report.append(f"The class '{worst_f1_class}' has the lowest F1-score ({lowest_f1:.4f}).\n")
        report.append("Potential causes:\n")
        report.append("- The class may be underrepresented in the dataset.\n")
        report.append("- The model may struggle to differentiate it from similar classes.\n")
        report.append("- Misclassification may be occurring due to overlapping features with another class.\n")

    report.append("\nRecommendations for Improvement:\n")
    report.append("- Adjust hyperparameters to optimize precision and recall balance.\n")
    report.append("- Apply data augmentation techniques to address class imbalances.\n")
    report.append("- Analyze misclassified samples to identify patterns of failure.\n")
    report.append("- Utilize a confusion matrix to better understand misclassification trends.\n")

    # Save to file
    with open(output_file, "w") as f:
        f.writelines(report)

    print(f"Analysis saved to {output_file}")


# Example Usage:
if __name__ == "__main__":
    analyze_metrics("metrics.json")
