import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageClassificationDataset(Dataset):
    """Custom Dataset for loading image classification data"""

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (string): Directory with all the images organized in train/test/val folders
            split (string): 'train', 'test', or 'val' split
            transform (callable, optional): Optional transform to be applied to samples
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform

        # Get all classes (assuming folder names are class names)
        self.classes = [d for d in os.listdir(self.data_dir)
                        if os.path.isdir(os.path.join(self.data_dir, d))]
        self.classes.sort()  # Sort to ensure consistent class indices

        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label


# Example usage
def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for train, test, and validation sets

    Args:
        data_dir (string): Root directory with train/test/val folders
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker threads for loading data

    Returns:
        dict: Dictionary containing train, test, and val data loaders
    """
    # Define transformations
    # For training - with augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # For validation and testing - only resize and normalize
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ImageClassificationDataset(data_dir, split='train', transform=train_transform)
    val_dataset = ImageClassificationDataset(data_dir, split='validation', transform=eval_transform)
    test_dataset = ImageClassificationDataset(data_dir, split='test', transform=eval_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Get class names for reference
    class_names = train_dataset.classes

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_names': class_names
    }


# Example of how to use the data loaders
if __name__ == "__main__":
    # Assuming your data is organized as:
    # data_root/
    #   ├── train/
    #   │   ├── class1/
    #   │   │   ├── img1.jpg
    #   │   │   └── ...
    #   │   └── class2/
    #   │       ├── img1.jpg
    #   │       └── ...
    #   ├── val/
    #   │   ├── class1/
    #   │   └── class2/
    #   └── test/
    #       ├── class1/
    #       └── class2/

    data_dir = "path/to/your/data"
    loaders = get_data_loaders(data_dir, batch_size=32)

    # Print dataset information
    train_loader = loaders['train']
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of classes: {len(loaders['class_names'])}")
    print(f"Class names: {loaders['class_names']}")

    # Get a batch of images
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")