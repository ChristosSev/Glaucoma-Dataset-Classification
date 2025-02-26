import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.models as models
from training import train_model
from evaluation import evaluate_model
from visualization import plot_training_history, plot_confusion_matrix
from shap import explain_model

def main(data_dir, batch_size=32, num_epochs=5, learning_rate=0.0001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    from Dataloader import get_data_loaders
    loaders = get_data_loaders(data_dir, batch_size=batch_size)

    # Get class information
    num_classes = len(loaders['class_names'])
    class_names = loaders['class_names']
    print(f"Training on {num_classes} classes: {class_names}")

    # Load pre-trained ResNet model
    model = models.resnet18(pretrained=True)


    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)


    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print("Starting training...")
    model, history = train_model(
        model,
        {'train': loaders['train'], 'val': loaders['val']},
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate_model(
        model,
        loaders['test'],
        criterion,
        device=device
    )

    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, class_names=class_names)

    # SHAP Explanation
    print("Generating SHAP explanations...")
    shap_values = explain_model(model, loaders['test'], device, num_samples=100)


    # Create directory for saved models if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save model
    torch.save(model.state_dict(), 'models/resnet_classifier.pth')
    print("Model saved as 'models/resnet_classifier.pth'")

    return model, history, (test_loss, test_acc)

if __name__ == "__main__":
    # Path to your dataset
    data_dir = "./release-raw/release-raw"

    # Run training pipeline
    model, history, test_metrics = main(
        data_dir=data_dir,
        batch_size=16,
        num_epochs=5,
        learning_rate=0.0001
    )
