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
from sklearn.metrics import accuracy_score
import time

#load train/test data into memory, on GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor()
])

# Relative path to the dataset
relative_dataset_path = 'Colorectal Cancer'

# Convert the relative path to an absolute path
absolute_dataset_path = os.path.abspath(relative_dataset_path)

# Use the absolute path in ImageFolder
dataset = ImageFolder(root=absolute_dataset_path, transform=transform)

#Split data into training, validation and test sets
print("Splitting data into train, test, and validation sets...")
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
print("done.")

#hyperparameters
learning_rates = [0.1, 0.001, 0.0005]
batch_sizes = [32, 64, 128]
num_epochs = 5

best_val_accuracy = 0
best_hyperparams = {}
best_model = None

print("training loop starting...")
for learning_rate in learning_rates:
    print(f"learning rate: {learning_rate}")
    for batch_size in batch_sizes:
        print(f"batch_size: {batch_size}")

        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        model = Resnet18(num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Training loop with validation
        for epoch in range(num_epochs):
            start_time = time.time()
            print(f"Epoch: {epoch}")
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device) #move data to gpu
                pred = model(images)
                loss = criterion(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            scheduler.step()

            train_time = time.time() - start_time

            # Validation loop
            start_time = time.time()
            model.eval()
            val_labels = []
            val_preds = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device) #move data to gpu
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    val_labels.extend(labels.numpy())
                    val_preds.extend(predicted.numpy())
            val_time = time.time() - start_time

            val_accuracy = accuracy_score(val_labels, val_preds)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s')

            # Save the best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_hyperparams = {'learning_rate': learning_rate, 'batch_size': batch_size}
                best_model = model


# Print the best hyperparameters
print(f'Best Hyperparameters: Learning Rate: {best_hyperparams["learning_rate"]}, Batch Size: {best_hyperparams["batch_size"]}')

# Evaluate the best model on the test set
best_model.eval()
test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = best_model(images)
        _, predicted = torch.max(outputs, 1)
        test_labels.extend(labels.numpy())
        test_preds.extend(predicted.numpy())

#print the classification report
test_report = classification_report(test_labels, test_preds)
print(test_report)

# Save the best model
best_model = best_model.to('cpu') #move model back to cpu
torch.save(best_model, 'Best_ColorectalCancerClassifier.pth')