import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import torch.optim as optim
import argparse
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from metrics_int import *
from train_utils import hyperparameter_search
from tsne import tsne_visualization
import json
from visualization import *
from torch.utils.data import DataLoader
from training import train_model
from evaluation import evaluate_model
from Gradcam import grad_cam_explanation
import sys


sys.argv = sys.argv[:1]  # Only keep the first element


def get_data_loaders(data_dir, batch_size=32, use_augmentation=False):
    print(f"Loading data from {data_dir}")

    if use_augmentation:
        # Training transforms with augmentations
        print("Using data augmentation for training.")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Simple transforms without augmentation
        print("No data augmentation for training.")
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Validation/Test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/validation", transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=eval_transform)

    print(f"Found {len(train_dataset)} training samples.")
    print(f"Found {len(val_dataset)} validation samples.")
    print(f"Found {len(test_dataset)} test samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_names': train_dataset.classes
    }


def main(data_dir, batch_size=32, num_epochs=15, learning_rate=0.001, optimizer_type='adam', weight_decay=0.0001,
         use_augmentation=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = get_data_loaders(data_dir, batch_size=batch_size, use_augmentation=use_augmentation)
    num_classes = len(loaders['class_names'])
    class_names = loaders['class_names']
    print(f"Training on {num_classes} classes: {class_names}")

    model = efficientnet_b0(pretrained=True)
    #print(f"Initial model architecture: {model}")

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)

    # Binary Cross-Entropy Loss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    #print(f"Loss function: {criterion}")

    # Setup optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    #print(f"Optimizer setup: {optimizer}")

    print("Starting training...")
    model, history = train_model(
        model,
        {'train': loaders['train'], 'val': loaders['val']},
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        patience=5
    )
    print(f"Training history: {history}")

    print("Evaluating on test set...")
    test_loss, test_acc, precision, recall, f1, roc_auc, per_class_metrics = evaluate_model(
        model,
        loaders['test'],
        criterion,
        device=device
    )

    print(
        f"Test Accuracy: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {roc_auc:.4f}")

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "per_class_metrics": per_class_metrics
    }
    metrics_file = "test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Generating model analysis report")
    analysis_report = analyze_metrics(metrics_file, "model_analysis.txt")
    print("Generated model analysis report")

    print("Generating Grad-CAM explanations...")
    # Get a sample image for visualization
    test_image = next(iter(loaders['test']))[0][0].to(device)

    # Get prediction - for binary classification
    with torch.no_grad():
        output = model(test_image.unsqueeze(0))
        predicted_class = int(output.item() > 0.5)  # Convert to 0 or 1

    # Generate Grad-CAM visualization
    overlay, cam = grad_cam_explanation(model, test_image.cpu(), predicted_class, device)

    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.title(f"Grad-CAM for class: {class_names[predicted_class]}")
    plt.savefig("gradcam_explanation.png")
    plt.close()

    # Generate t-SNE visualization
    print("Generating t-SNE visualization...")
    tsne_visualization(model, loaders['test'], device)

    print("Plotting Confusion Matrix...")
    # Get predictions and true labels for confusion matrix
    true_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # Binary classification threshold
            preds = (outputs > 0.5).float().squeeze()
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, pred_labels, class_names=class_names)

    return model, history, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binary Classification Training')
    parser.add_argument('--data_dir', type=str, default='./release-raw/release-raw/', help='Path to dataset')
    parser.add_argument('--use_hyperparam_search', action='store_true', help='Run hyperparameter search')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation for training')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--hp_epochs', type=int, nargs='+', default=[20, 30, 40],
                        help='Epochs to try in hyperparameter search')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')

    args = parser.parse_args()

    # Print all arguments
    print("Training with the following parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    if args.use_hyperparam_search:
        print("Running hyperparameter search...")
        best_params = hyperparameter_search(
            args.data_dir,
            hp_epochs=args.hp_epochs,
            use_augmentation=args.use_augmentation
        )
        print("Training with best hyperparameters...")
        model, history, test_metrics = main(
            data_dir=args.data_dir,
            batch_size=best_params['batch_size'],
            num_epochs=best_params['epochs'],
            learning_rate=best_params['lr'],
            optimizer_type=best_params['optimizer'],
            weight_decay=best_params['weight_decay'],
            use_augmentation=args.use_augmentation
        )
    else:
        print("Training with default hyperparameters...")
        model, history, test_metrics = main(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer,
            weight_decay=args.weight_decay,
            use_augmentation=args.use_augmentation
        )

    print(f"Final test accuracy: {test_metrics['test_accuracy']:.4f}")
