import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import torchvision.models as models
import itertools
import matplotlib.pyplot as plt
from utils.training import train_model
from utils.evaluation import evaluate_model
from Gradcam import grad_cam_explanation
from train_utils import *

def main(data_dir, batch_size=32, num_epochs=1, learning_rate=0.0001, optimizer_type='adam', weight_decay=0):
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
    model = model.to(device)  # Move the model to the correct device

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    # Train model
    print("Starting training...")
    model, history = train_model(
        model,
        {'train': loaders['train'], 'val': loaders['val']},
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        patience=5  # Add early stopping
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate_model(
        model,
        loaders['test'],
        criterion,
        device=device
    )

    # Grad-CAM
    print("Generating Grad-CAM explanations...")
    # Example: Use the first test image and the predicted class
    test_image, test_label = loaders['test'].dataset[0]
    predicted_class = test_preds[0]

    overlay, cam = grad_cam_explanation(model, test_image, predicted_class, device)

    # Plot the Grad-CAM result
    plt.imshow(overlay)
    plt.title(f"Grad-CAM for class: {class_names[predicted_class]}")
    plt.show()

    return model, history, (test_loss, test_acc)


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Binary Classficitaion ')
    parser.add_argument('--data_dir', type=str, default='./release-raw/release-raw',
                        help='Path to dataset')
    parser.add_argument('--use_hyperparam_search', action='store_true',
                        help='Run hyperparameter search')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (used when not doing hyperparameter search)')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs (used when not doing hyperparameter search)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (used when not doing hyperparameter search)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer type (used when not doing hyperparameter search)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (used when not doing hyperparameter search)')

    args = parser.parse_args()

    # Path to your dataset
    data_dir = args.data_dir

    if args.use_hyperparam_search:
        print("Running hyperparameter search...")
        best_params = hyperparameter_search(data_dir)

        print("Training with best hyperparameters...")
        model, history, test_metrics = main(
            data_dir=data_dir,
            batch_size=best_params['batch_size'],
            num_epochs=best_params['epochs'],
            learning_rate=best_params['lr'],
            optimizer_type=best_params['optimizer'],
            weight_decay=best_params['weight_decay']
        )
    else:
        print("Training with default hyperparameters...")
        model, history, test_metrics = main(
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer,
            weight_decay=args.weight_decay
        )

    print(f"Final test accuracy: {test_metrics[1]}")
