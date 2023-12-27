import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import zipfile
import os
from models import Resnet18
from tools import extract_features, plot_with_labels
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np


# Load pre-trained model
model = torch.load('ColorectalCancerClassifier.pth')
model.eval()  # set model to eval mode

transform = transforms.Compose([
    transforms.ToTensor()
])

# Relative path to the dataset
relative_dataset_path = 'Animal Faces'

print("Starting...")

# Convert the relative path to an absolute path
absolute_dataset_path = os.path.abspath(relative_dataset_path)

# Use the absolute path in ImageFolder
dataset = ImageFolder(root=absolute_dataset_path, transform=transform)

# Split data into training and test sets
print("Splitting data...")
train_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.3])
print("done...")

batch_size = 64

train_loader = DataLoader(train_set, shuffle=True,
                          batch_size=batch_size)  # create train loader
test_loader = DataLoader(test_set, batch_size=batch_size)  # create test loader

# Extract features from train data using your custom trained model
print("extracting features...")
train_features, train_labels = extract_features(
    model, train_loader, is_custom_model=True)

# Extract features from test data using your custom trained model
test_features, test_labels = extract_features(
    model, test_loader, is_custom_model=True)
print("done...")


# Apply T-SNE to train and test data
print("T-SNE...")
train_tsne = TSNE(n_components=2, random_state=123)
train_reduced_features = train_tsne.fit_transform(train_features)

test_tsne = TSNE(n_components=2, random_state=123)
test_reduced_features = test_tsne.fit_transform(test_features)
print("Done...")

# Plot train data
plot_with_labels(train_reduced_features, train_labels,
                 "Train T-SNE for Dataset 3 using Task 1 Encoder")

# Plot test data
plot_with_labels(test_reduced_features, test_labels,
                 "Test T_SNE for Dataset 3 using Task 1 Encoder")


# Convert features and labels for scikit-learn
X_train = train_features
y_train = np.array(train_labels)
X_test = test_features
y_test = np.array(test_labels)

# Initialize KNN classifier
print("training knn classifier...")
knn = KNeighborsClassifier(n_neighbors=3)

# Fit model on train set
knn.fit(X_train, y_train)
print("done...")

# Predict labels on test set
y_pred = knn.predict(X_test)

# Evaluate model
with open('KNN_Encoder1.txt', 'w') as file:
    file.write('KNN Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\nKNN Classification Report:\n')
    file.write(classification_report(y_test, y_pred))
    file.write('\nKNN Accuracy: ')
    file.write(str(accuracy_score(y_test, y_pred)))

print('KNN Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('KNN Classification Report:\n', classification_report(y_test, y_pred))
print('KNN Accuracy:', accuracy_score(y_test, y_pred))

#save knn model
torch.save(knn, "KNeighborsClassifier.pth")


# Initialize SVM classifier
svm_classifier = SVC(kernel='rbf')

# Fit model on train set
svm_classifier.fit(X_train, y_train)

# Predict labels on test set
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate model

# Open a file in write mode
with open('SVM_Encoder1.txt', 'w') as file:
    file.write('SVM Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred_svm)))
    file.write('\n\nSVM Classification Report:\n')
    file.write(classification_report(y_test, y_pred_svm))
    file.write('\nSVM Accuracy:')
    file.write(str(accuracy_score(y_test, y_pred_svm)))

print('SVM Confusion Matrix:\n', confusion_matrix(y_test, y_pred_svm))
print('SVM Classification Report:\n', classification_report(y_test, y_pred_svm))
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))


#save svm model
torch.save(svm_classifier, 'SVMClassifier.pth')