# CNN-Based Classification Projects

This repository contains two different classification projects implemented using Convolutional Neural Networks (CNN). Both projects are developed using TensorFlow/Keras.

## 📁 Project Structure

```
CNN/
├── 01-Malaria_tensorflow_CNN/
│   ├── Malaria_CNN (1).ipynb
│   ├── saved_model.pb
│   ├── keras_metadata.pb
│   └── variables.index
├── 02-Brain_Tumor_Classification/
│   ├── brain-tumor-classification-mri-cnn.ipynb
│   └── brain-tumor-vgg16.ipynb
└── README.md
```

## 01-Malaria_tensorflow_CNN

### Project Description
A CNN model developed for detecting malaria parasites in blood cells. This project is used to classify infected and non-infected blood cells.

### Features
- **Data Source**: Malaria dataset from TensorFlow Datasets
- **Model**: Custom CNN architecture
- **Classification**: Binary (Infected/Non-infected)
- **Data Split**: 60% Training, 20% Validation, 20% Test

### Technical Details
- **Framework**: TensorFlow/Keras
- **Data Preprocessing**: Normalization and data augmentation
- **Model Architecture**: Conv2D, MaxPool2D, Dense layers
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

### Usage
1. Open `Malaria_CNN (1).ipynb` in Jupyter Notebook
2. Install required libraries
3. Run the notebook cell by cell

## 🧠 02-Brain_Tumor_Classification

### Project Description
CNN models developed for classifying brain tumor MRI images. This project is used to detect different types of brain tumors.

### Contents
1. **brain-tumor-classification-mri-cnn.ipynb**: Custom CNN architecture
2. **brain-tumor-vgg16.ipynb**: VGG16 transfer learning model

### Features
- **Data Source**: MRI brain tumor images
- **Classification**: 4 different tumor types
- **Data Augmentation**: Shear, zoom, horizontal flip
- **Image Size**: 64x64 pixels

### Technical Details
- **Framework**: TensorFlow/Keras
- **Data Preprocessing**: ImageDataGenerator with data augmentation
- **Model Types**: 
  - Custom CNN architecture
  - VGG16 transfer learning
- **Batch Size**: 32
- **Validation Split**: 10%

### Usage
1. Open the relevant notebook file in Jupyter
2. Place the dataset in the appropriate location
3. Run the notebook

##  Requirements

### Core Libraries
```bash
pip install tensorflow
pip install numpy
pip install matplotlib
pip install pandas
pip install tensorflow-datasets
```


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CNN
```


## Model Performance

### Malaria Detection
- **Accuracy**: High accuracy rate
- **Binary Classification**: Infected/Non-infected
- **Model Saving**: Trained model saved

### Brain Tumor Classification
- **Multi-class Classification**: 4 different tumor types
- **Transfer Learning**: VGG16 pre-trained model
- **Data Augmentation**: Image variety enhancement

## Customization

### Dataset Modification
- Different medical imaging datasets for malaria project
- Different MRI datasets for brain tumor project

### Model Parameters
- Learning rate adjustment
- Batch size modification
- Increasing/decreasing epoch count

## Notes

- Both projects are compatible with TensorFlow 2.x
- GPU usage is recommended (for acceleration)
- Sufficient RAM required for large datasets
