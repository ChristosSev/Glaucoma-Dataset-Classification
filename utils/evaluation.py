from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_curve, auc
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0.0  # Initialize loss variable

    # For ROC curve
    all_true_labels = []
    all_scores = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Calculate loss
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            # For ROC curve
            all_true_labels.append(labels.cpu().numpy())
            all_scores.append(probs.cpu().numpy()[:, 1])  # Assuming binary classification

    test_loss /= len(data_loader)  # Average loss over all batches

    # Convert to numpy arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_true_labels = np.concatenate(all_true_labels)
    all_scores = np.concatenate(all_scores)

    # Overall metrics
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Per-class metrics
    per_class_report = classification_report(all_labels, all_preds, output_dict=True)
    per_class_metrics = {str(i): per_class_report[str(i)] for i in range(len(per_class_report)-3)}

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_true_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')  # Save the ROC curve plot
    plt.close()

    # Save metrics to a JSON file
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "per_class": per_class_metrics
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    return test_loss, accuracy, precision, recall, f1, roc_auc, per_class_metrics
