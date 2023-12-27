import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import zipfile
import os
from models import Resnet18
from tools import extract_features, plot_with_labels
from sklearn.manifold import TSNE


# Load pre-trained model
model = torch.load('ColorectalCancerClassifier.pth')
model.eval()  # set model to eval mode

transform = transforms.Compose([
    transforms.ToTensor()
])

# Relative path to the dataset
relative_dataset_path = 'Colorectal Cancer'

# Convert the relative path to an absolute path
absolute_dataset_path = os.path.abspath(relative_dataset_path)

# Use the absolute path in ImageFolder
dataset = ImageFolder(root=absolute_dataset_path, transform=transform)

# Split data into training and test sets
train_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.3])

learning_rate = 0.001
num_epochs = 10
batch_size = 64

train_loader = DataLoader(train_set, shuffle=True,
                          batch_size=batch_size)  # create train loader
test_loader = DataLoader(test_set, batch_size=batch_size)  # create test loader

# Extract features from train data
train_features, train_labels = extract_features(
    model, train_loader, is_custom_model=True)

# Extract features from test data
test_features, test_labels = extract_features(
    model, test_loader, is_custom_model=True)

# Apply T-SNE to train and test data
train_tsne = TSNE(n_components=2, random_state=123)
train_reduced_features = train_tsne.fit_transform(train_features)

test_tsne = TSNE(n_components=2, random_state=123)
test_reduced_features = test_tsne.fit_transform(test_features)

# Plot train data
plot_with_labels(train_reduced_features, train_labels, "Train T-SNE")

# Plot test data
plot_with_labels(test_reduced_features, test_labels, "Test T_SNE")
