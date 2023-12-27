import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import zipfile

zip_file = 'Dataset 1.zip'
dir = ''
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
  print("Extracting data...")
  zip_ref.extractall(dir)
  print("Done!")


zip_file = 'Dataset 2.zip'
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
  print("Extracting data...")
  zip_ref.extractall(dir)
  print("Done!")


zip_file = 'Dataset 3.zip'
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
  print("Extracting data...")
  zip_ref.extractall(dir)
  print("Done!")