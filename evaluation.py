import torch
from tqdm import tqdm


def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    Evaluate the model on a test dataset

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test dataset
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)

    Returns:
        test_loss: Average loss on test set
        test_acc: Accuracy on test set
        all_preds: List of all predictions
        all_labels: List of all true labels
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # Store predictions and true labels for confusion matrix
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Save for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate test statistics
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    return test_loss, test_acc, all_preds, all_labels