from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

def tsne_visualization(model, data_loader, device, filename='tsne_plot.png'):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            features = outputs.cpu().numpy()  # Use the model outputs as features
            all_features.append(features)
            all_labels.append(labels.cpu().numpy())

    # Convert to numpy arrays
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    # Apply t-SNE to reduce the dimensions of the feature vectors
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(all_features)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Model Outputs')
    plt.savefig(filename)  # Save the plot as an image
    plt.close()

