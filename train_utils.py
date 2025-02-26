import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import itertools
from utils.training import train_model
from utils.evaluation import evaluate_model
from utils.visualization import plot_training_history, plot_confusion_matrix
from utils.evaluation import evaluate_model
from metrics_int import analyze_metrics

# Import data loader
from Dataloader import get_data_loaders


def create_model(num_classes, device):
    """Create a ResNet model with the correct number of output classes."""
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def get_optimizer(model, optimizer_type, learning_rate, weight_decay):
    """Initialize optimizer based on type."""
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type")


def train_and_evaluate(data_dir, batch_size, num_epochs, learning_rate, optimizer_type, weight_decay):
    """Train and evaluate model with given hyperparameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_data_loaders(data_dir, batch_size=batch_size)

    num_classes = len(loaders['class_names'])
    model = create_model(num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_type, learning_rate, weight_decay)

    print(
        f"Starting training with lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}, optimizer={optimizer_type}, wd={weight_decay}")
    model, history = train_model(model, {'train': loaders['train'], 'val': loaders['val']}, criterion, optimizer,
                                 num_epochs, device)

    print("Evaluating model...")
    test_loss, test_acc, precision, recall, f1_score, per_class_metrics = evaluate_model(model, loaders['test'], criterion, device)

    # Analyze the metrics
    analyze_metrics('metrics.json')  # This will create the analysis report
    print(
        f"Returning: test_loss={test_loss}, accuracy={test_acc}, precision={precision}, recall={recall}, f1_score={f1_score}, per_class_metrics={per_class_metrics}")

    return test_loss, test_acc, precision, recall, f1_score, per_class_metrics



def hyperparameter_search(data_dir):
    """Perform grid search over hyperparameters."""
    learning_rates = [0.001, 0.01, 0.0001]
    batch_sizes = [16, 32, 64]
    epochs = [1, 5]
    optimizers = ['adam', 'sgd']
    weight_decays = [0, 0.0001, 0.001]

    best_acc = 0
    best_params = {}

    for lr, batch_size, epoch, optimizer, wd in itertools.product(learning_rates, batch_sizes, epochs, optimizers,
                                                                  weight_decays):
        test_loss, test_acc, precision, recall, f1_score, per_class_metrics = train_and_evaluate(
            data_dir, batch_size, epoch, lr, optimizer, wd
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_params = {'lr': lr, 'batch_size': batch_size, 'epochs': epoch, 'optimizer': optimizer,
                           'weight_decay': wd}

    print(f"Best accuracy: {best_acc}")
    print(f"Best parameters: {best_params}")
    return best_params

