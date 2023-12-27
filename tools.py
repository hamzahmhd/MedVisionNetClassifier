import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from models import Resnet18
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def extract_features(model, dataloader, is_custom_model=False):
    features = []
    labels = []

    with torch.no_grad():
        for inputs, label in dataloader:
            if is_custom_model:
                feature = model(inputs, feature_extract=True)
            else:
                feature = model(inputs)
            features.append(feature)

            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            labels.append(label)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return features, labels


def plot_with_labels(low_dim_embs, labels, title):
    plt.figure(figsize=(12, 12))
    plt.title(title)
    unique_labels = set(labels)
    for label in unique_labels:
        x = low_dim_embs[labels == label, 0]
        y = low_dim_embs[labels == label, 1]
        plt.scatter(x, y, label=label)
    plt.legend()
    plt.show()
