# CNNs and Transfer Learning for Biomedical Image Classification

## Project Description

This project explores the practical applications of computer vision by leveraging convolutional neural networks (CNNs) for enhanced image classification, with a specific focus on computational pathology. It also encompasses a study on transfer learning and feature extraction. The objectives are:

1. To train a custom ResNet-18 CNN model for classifying Colorectal Cancer.
2. To utilize the encoder from this custom model and a pre-trained ImageNet model for feature extraction on Prostate Cancer and Animal Faces datasets using t-SNE.
3. To classify the extracted features using a K-nearest neighbors (KNN) clustering model and a Support Vector Machine (SVM) model, followed by a comprehensive analysis.

## Setup Instructions

### Environment Setup

1. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate
```

2. Install the required packages:

```
pip install -r requirements.txt
```

### Data Preparation

- Download the dataset zip files (datasets 1-3). Note: The datasets are not included in the GitHub repository due to their size.
- Rename the files according to the naming conventions in `data_utils.py`.
- Extract the images for each dataset:

```
python data_utils.py
```

## Executing Tasks

### Task 1: Colorectal Cancer Classification

1. Train and validate the model:

```
python task1.py
```

2. Analyze the results:

```
python analysis.py
```

### Task 2: Feature Extraction and Classification

1. Run feature extraction on datasets using the encoders:

```
python encoder1_data2.py
python encoder1_data3.py
python encoder2_data2.py
python encoder2_data3.py
```

These scripts will train and validate the models, saving the best-performing models as `.pth` files.

## Running the Sample Data

- To test the Colorectal Dataset trained ResNet-18 model on a sample dataset, run:
- Note: add the dataset number at the end of command to run samples from that dataset

```
python sample.py 1
python sample.py 2
python sample.py 3
```

- To use your own sample data, replace `sample.zip`

## Downloading project datasets

[Dataset 1] ([https://1drv.ms/u/s!AilzKc-njjP7mN0LqoRZvUYONY9sbQ?e=wxWbip](https://1drv.ms/u/s!AilzKc-njjP7mN0NOZvxl0TPAUxmig?e=K0TpeX)https://1drv.ms/u/s!AilzKc-njjP7mN0NOZvxl0TPAUxmig?e=K0TpeX)

[Dataset 2](https://1drv.ms/u/s!AilzKc-njjP7mN0M_LjB5xeAydDsrA?e=0obzsx)

[Dataset 3](https://1drv.ms/u/s!AilzKc-njjP7mN0LqoRZvUYONY9sbQ?e=wxWbip)
