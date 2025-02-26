import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models

# Hook to get the gradients
class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        # Hook the target layer to capture feature maps and gradients
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class):
        # Perform a forward pass
        self.model.eval()
        input_image.requires_grad = True
        output = self.model(input_image)

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass to get gradients for the target class
        class_score = output[0, target_class]
        class_score.backward()

        # Get the gradients and feature maps
        gradients = self.gradients[0].cpu().data.numpy()
        feature_maps = self.feature_maps[0].cpu().data.numpy()

        # Pool the gradients across the feature maps
        weights = np.mean(gradients, axis=(1, 2))  # Global Average Pooling (spatial dimensions)

        # Compute the Grad-CAM
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i, :, :]

        # Apply ReLU to the Grad-CAM
        cam = np.maximum(cam, 0)

        # Normalize the CAM
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam

    def overlay_heatmap(self, image, cam):
        # Convert the input image from Tensor to numpy for overlaying
        image = image.cpu().data.numpy().transpose(1, 2, 0)
        image = np.uint8(255 * image)

        # Create a heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # Overlay the heatmap on the original image
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

        return overlay

def grad_cam_explanation(model, image, target_class, device):
    # Ensure the image is on the same device as the model
    image = image.to(device)  # Move the image to GPU or CPU based on the device

    # Choose the last convolutional layer (for ResNet, it's 'layer4')
    target_layer = model.layer4[1].conv2  # Use the second convolution layer in the last block
    grad_cam = GradCam(model, target_layer)

    # Generate Grad-CAM heatmap
    cam = grad_cam.generate_cam(image.unsqueeze(0), target_class)

    # Overlay the heatmap on the image
    overlay = grad_cam.overlay_heatmap(image, cam)

    return overlay, cam
