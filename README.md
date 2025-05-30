# 🌌 Galaxy Morphology Classification Using Deep Learning

This project classifies galaxy images into **Spiral**, **Lenticular**, and **Elliptical** types using deep learning models. We designed a custom CNN and compared it with a pre-trained **ResNet50** architecture, leveraging **Keras** and **TensorFlow** frameworks.

## 🧠 Project Overview

- **Goal**: Automatically identify galaxy morphology from image data.
- **Models Used**:
  - Custom Convolutional Neural Network (CNN)
  - ResNet50 with Transfer Learning
- **Dataset**: 10,000 labeled galaxy images categorized into 3 classes.

## 🗃️ Dataset

- **Source**: Galaxy morphology dataset (10,000 images)
- link : https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images
- **Classes**:
  - Spiral
  - Lenticular
  - Elliptical
- **Splits**:
  - Train: 6,833 images
  - Validation: 3,450 images
  - Test: 450 images

### 🔧 Preprocessing
- Resizing all images to `224x224`
- Normalizing pixel values
- Data augmentation (rescaling, shear, zoom, horizontal flip)
- Used `ImageDataGenerator` and Keras' `flow_from_directory`

## 🏗️ Model Architectures

### 🔹 Custom CNN
- 4 Convolutional layers with filters: 32, 64, 128, 256
- Batch Normalization & Max Pooling
- Global Average Pooling
- Dense layer + Dropout
- Output: 3-class Softmax

### 🔹 ResNet50 (Transfer Learning)
- Pretrained on ImageNet
- Feature extractor frozen
- Custom classifier head with:
- Global Average Pooling
- Dense → Dropout → Softmax (3 classes)

## 📈 Training & Evaluation

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Callbacks**:
  - EarlyStopping
  - ReduceLROnPlateau
- **Evaluation Metric**: Accuracy

### 🔬 Results
| Model     | Accuracy |
|-----------|----------|
| Custom CNN | Moderate |
| ResNet50   | High     |

ResNet50 outperformed the custom model due to better feature extraction and deeper architecture.

## ⚠️ Challenges
- Overfitting despite dropout and augmentation
- Imbalanced dataset among galaxy classes
- High computational demand during training

## ✅ Conclusion

This project demonstrates how deep learning can effectively classify galaxy morphologies. The **ResNet50** model yielded the best results. Future improvements could involve:
- Using more balanced and larger datasets
- Fine-tuning pre-trained models
- Exploring hybrid or ensemble approaches

## 📂 Project Structure
├── data/ # Raw image data 
├── notebooks/
│ └── DL_GalaxyMorphology.ipynb # Jupyter notebook with code and results
├── Report_GalaxyMorphology.pdf # Detailed technical report
├── README.md # Project documentation


## 👨‍💻 Authors

- **Abdel Aziz Elhelaly** – ID: 202201827  
- **Ezz Eldeen** – ID: 202202132  

---
