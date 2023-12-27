import sys
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from models import Resnet18  # Adjust this import based on your project structure
import zipfile

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained model
pretrained_model_path = "ColorectalCancerClassifier.pth"
model = torch.load(pretrained_model_path)
model.eval()

#Get user argument
if len(sys.argv) < 2:
    print("Usage: python sample.py <option_number>")
    sys.exit(1)

option = sys.argv[1]

if option == '1':
    print("using sample dataset 1: Colorectal Cancer")
    zip_file_path = 'sample1.zip'
    root_dir = "sample_data/Colorectal Cancer"
elif option == '2':
    print("using sample dataset 2: Prostate Cancer")
    zip_file_path = 'sample2.zip'
    root_dir = "sample_data/Prostate Cancer"
elif option == '3':
    zip_file_path = 'sample3.zip'
    print("using sample dataset 3: Animal Faces ")
    root_dir = "sample_data/Animal Faces"
else:
    print("Invalid option. Please choose 1, 2, or 3.")
    sys.exit(1)

# Load data from the ZIP file
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall("sample_data")

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Create a DataLoader for the inference data
inference_dataset = ImageFolder(root=root_dir, transform=transform)
inference_loader = DataLoader(inference_dataset, batch_size=64, shuffle=False)

# Perform inference on the data
predictions = []
ground_truth = []

with torch.no_grad():
    for images, labels in inference_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        ground_truth.extend(labels.numpy())

# Calculate and print the classification report
classification_report_str = classification_report(ground_truth, predictions)
print(classification_report_str)
