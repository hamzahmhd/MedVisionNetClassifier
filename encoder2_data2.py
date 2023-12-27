import torch
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tools import extract_features, plot_with_labels
from sklearn.manifold import TSNE
import os

# Define module to bypass fully connected layer


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


# Load pre-trained ImageNet model
pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
pretrained_model.fc = Identity()

# Set model to eval mode
pretrained_model.eval()

# Preprocessing of data according to ImageNet's training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Relative path to the dataset
relative_dataset_path = 'Prostate Cancer'

# Convert the relative path to an absolute path
absolute_dataset_path = os.path.abspath(relative_dataset_path)

# Use the absolute path in ImageFolder
dataset = ImageFolder(root=absolute_dataset_path, transform=transform)

# Split data into training and test sets
train_set, test_set = torch.utils.data.random_split(
    dataset, [int(0.7 * len(dataset)), len(dataset) - int(0.7 * len(dataset))])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Extract features using the pretrained model
train_features, train_labels = extract_features(pretrained_model, train_loader)
test_features, test_labels = extract_features(pretrained_model, test_loader)

# Apply T-SNE to train and test features
train_tsne = TSNE(n_components=2, random_state=123).fit_transform(
    train_features)
test_tsne = TSNE(n_components=2, random_state=123).fit_transform(test_features)

# Plot the T-SNE results
plot_with_labels(train_tsne, train_labels,
                 "Train T-SNE for Dataset 2 using Pretrained ImageNet Encoder")
plot_with_labels(test_tsne, test_labels,
                 "Test T_SNE for Dataset 2 using Pretrained ImageNet Encoder")
