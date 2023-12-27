import torch
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tools import extract_features, plot_with_labels
from sklearn.manifold import TSNE
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
relative_dataset_path = 'Animal Faces'

# Convert the relative path to an absolute path
absolute_dataset_path = os.path.abspath(relative_dataset_path)

# Use the absolute path in ImageFolder
dataset = ImageFolder(root=absolute_dataset_path, transform=transform)

print("Beginning training...")
print("Splitting data into training and test sets")
# Split data into training and test sets
train_set, test_set = torch.utils.data.random_split(
    dataset, [int(0.7 * len(dataset)), len(dataset) - int(0.7 * len(dataset))])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Extract features using the pretrained model
print("Extracting features...")
train_features, train_labels = extract_features(pretrained_model, train_loader)
test_features, test_labels = extract_features(pretrained_model, test_loader)

# Apply T-SNE to train and test features
print("T-SNE train...")
train_tsne = TSNE(n_components=2, random_state=123).fit_transform(
    train_features)
print("T-SNE test...")
test_tsne = TSNE(n_components=2, random_state=123).fit_transform(test_features)

# Plot the T-SNE results
plot_with_labels(train_tsne, train_labels,
                 "Train T-SNE for Dataset 3 using Pretrained ImageNet Encoder")
plot_with_labels(test_tsne, test_labels,
                 "Test T_SNE for Dataset 3 using Pretrained ImageNet Encoder")


# Convert features and labels for scikit-learn
X_train = train_features
y_train = np.array(train_labels)
X_test = test_features
y_test = np.array(test_labels)


# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit model on train set
print("Training knn classifier...")
knn.fit(X_train, y_train)
print("done...")

# Predict labels on test set
y_pred = knn.predict(X_test)

# Evaluate model
with open('KNN_Encoder2.txt', 'w') as file:
    file.write('KNN Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred)))
    file.write('\n\nKNN Classification Report:\n')
    file.write(classification_report(y_test, y_pred))
    file.write('\nKNN Accuracy: ')
    file.write(str(accuracy_score(y_test, y_pred)))

print('KNN Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('KNN Classification Report:\n', classification_report(y_test, y_pred))
print('KNN Accuracy:', accuracy_score(y_test, y_pred))


# Initialize SVM classifier
svm_classifier = SVC(kernel='rbf')

# Fit model on train set
svm_classifier.fit(X_train, y_train)

# Predict labels on test set
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate model

with open('SVM_Encoder2.txt', 'w') as file:
    file.write('SVM Confusion Matrix:\n')
    file.write(str(confusion_matrix(y_test, y_pred_svm)))
    file.write('\n\nSVM Classification Report:\n')
    file.write(classification_report(y_test, y_pred_svm))
    file.write('\nSVM Accuracy:')
    file.write(str(accuracy_score(y_test, y_pred_svm)))

print('SVM Confusion Matrix:\n', confusion_matrix(y_test, y_pred_svm))
print('SVM Classification Report:\n', classification_report(y_test, y_pred_svm))
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))

torch.save(svm_classifier, 'SVMClassifierdata3.pth')
torch.save(knn, 'KNNdata3.pth')
