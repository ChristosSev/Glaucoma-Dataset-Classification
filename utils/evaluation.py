from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import json
import torch
import numpy as np

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0.0  # Initialize loss variable

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

    test_loss /= len(data_loader)  # Average loss over all batches

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Overall metrics
    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Per-class metrics
    per_class_report = classification_report(all_labels, all_preds, output_dict=True)
    per_class_metrics = {str(i): per_class_report[str(i)] for i in range(len(per_class_report)-3)}

    # Save metrics to a JSON file
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "per_class": per_class_metrics
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    return test_loss, accuracy, precision, recall, f1, per_class_metrics


