#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.getcwd())


# In[9]:


import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define transformations (convert to grayscale and normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 1-channel input
])

# Load datasets
train_dataset = ImageFolder(root='chest_xray/train', transform=transform)
val_dataset = ImageFolder(root='chest_xray/val', transform=transform)
test_dataset = ImageFolder(root='chest_xray/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check class mapping
print(f"Class Mapping: {train_dataset.classes}")  # Should print ['NORMAL', 'PNEUMONIA']


# In[10]:


from torchvision.models import resnet18
import torch.nn as nn

# Load ResNet18 architecture (no pretrained weights to avoid download issues)
resnet_model = resnet18(weights=None)  # Use weights=None for no pretrained weights

# Modify the first convolutional layer for 1-channel (grayscale) input
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Replace the fully connected layer for binary classification
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model.to(device)


# In[11]:


import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_accuracy = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_accuracy = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Train the ResNet model
train_model(resnet_model, train_loader, val_loader, epochs=10)


# In[12]:


from sklearn.metrics import classification_report, confusion_matrix

# Evaluate model on test set
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

# Evaluate the ResNet model
evaluate_model(resnet_model, test_loader)


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

def grad_cam(model, image, target_layer):
    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Variables to store gradients and features
    gradients = []
    features = []

    # Register hooks to capture gradients and features
    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks on the target layer
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    outputs = model(image)
    pred_class = outputs.argmax(dim=1).item()
    class_score = outputs[:, pred_class]

    # Backward pass
    model.zero_grad()
    class_score.backward()

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    # Process the captured gradients and features
    grad = gradients[0].cpu().detach()
    fmap = features[0].cpu().detach()

    # Weight the feature map by the mean of gradients
    weights = grad.mean(dim=[2, 3], keepdim=True)  # Global average pooling on gradients
    cam = (weights * fmap).sum(dim=1).squeeze()  # Weighted sum of feature maps

    # ReLU activation and normalization
    cam = torch.relu(cam)
    cam = cam / cam.max()

    # Upsample to match the input image size
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().numpy()
    return cam

# Example usage
example_image, _ = test_dataset[0]  # Take the first test sample
example_image = example_image.to(device)  # Send to device
target_layer = resnet_model.layer4  # Use the last convolutional block in ResNet

# Generate Grad-CAM heatmap
example_cam = grad_cam(resnet_model, example_image, target_layer)

# Visualize the result
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(example_image[0].cpu().numpy(), cmap='gray')  # Display grayscale image
plt.imshow(example_cam, cmap='jet', alpha=0.5)  # Overlay Grad-CAM heatmap
plt.title("Grad-CAM Visualization")
plt.axis('off')
plt.show()



# In[16]:


import os
import matplotlib.pyplot as plt

# Directory to save Grad-CAM visualizations
os.makedirs("grad_cam_outputs", exist_ok=True)

def save_grad_cam_images(model, data_loader, target_layer, output_dir="grad_cam_outputs"):
    model.eval()
    for idx, (image, label) in enumerate(data_loader):
        if idx > 10:  # Limit to first 10 images for demonstration
            break
        
        image = image[0].to(device)  # Get the first image in the batch
        label = label[0].item()  # Get the corresponding label
        cam = grad_cam(model, image, target_layer)  # Generate Grad-CAM
        
        # Save the visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(image[0].cpu().numpy(), cmap='gray')  # Original image
        plt.imshow(cam, cmap='jet', alpha=0.5)  # Grad-CAM overlay
        plt.title(f"Grad-CAM for Test Image {idx+1} (Label: {label})")
        plt.axis('off')
        plt.savefig(f"{output_dir}/grad_cam_{idx+1}.png")
        plt.close()

# Generate Grad-CAM for ResNet
save_grad_cam_images(resnet_model, test_loader, resnet_model.layer4)


# In[17]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(model, data_loader, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # Plot confusion matrix
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

# Plot confusion matrix for ResNet
plot_confusion_matrix(resnet_model, test_loader, classes=['NORMAL', 'PNEUMONIA'])


# In[18]:


def train_model_with_logging(model, optimizer, train_loader, val_loader, epochs):
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_accuracy = correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_accuracy = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies

# Train ResNet with logging
train_losses, train_accuracies, val_losses, val_accuracies = train_model_with_logging(
    resnet_model, resnet_optimizer, train_loader, val_loader, epochs=10
)


# In[19]:


def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# Plot curves
plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)


# In[20]:


def summarize_metrics(train_accuracies, val_accuracies, val_losses):
    final_train_accuracy = train_accuracies[-1]
    final_val_accuracy = val_accuracies[-1]
    final_val_loss = val_losses[-1]

    summary = {
        "Final Train Accuracy": final_train_accuracy,
        "Final Validation Accuracy": final_val_accuracy,
        "Final Validation Loss": final_val_loss
    }
    print("Performance Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

# Summarize metrics
summarize_metrics(train_accuracies, val_accuracies, val_losses)


# In[21]:


def analyze_misclassifications(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    images_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            images_list.extend(images.cpu())

    # Find misclassified indices
    misclassified_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]

    # Analyze Grad-CAM for misclassified examples
    for idx in misclassified_indices[:5]:  # Analyze first 5 misclassifications
        image = images_list[idx].unsqueeze(0).to(device)
        label = all_labels[idx]
        pred = all_preds[idx]

        cam = grad_cam(model, image[0], resnet_model.layer4)
        plt.figure(figsize=(8, 8))
        plt.imshow(image[0][0].cpu().numpy(), cmap='gray')
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title(f"True Label: {label}, Predicted: {pred}")
        plt.axis('off')
        plt.show()

# Analyze misclassifications for ResNet
analyze_misclassifications(resnet_model, test_loader)


# In[ ]:




