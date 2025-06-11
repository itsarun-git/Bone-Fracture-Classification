# 🦴 Fracture Classification with Deep Learning  
> Built using **TensorFlow**, **PyTorch**, **Keras**, and **Streamlit**

## 📌 Overview
This project uses deep learning techniques to classify different types of bone fractures from X-ray images. The aim is to explore and compare the performance of different model architectures and create a deployable app to test predictions.
---
## 📂 Project Structure

- `app.py`: Streamlit interface for uploading X-ray images and getting predictions.
- `fracture_classifier_final.keras`: Trained model file (**not included in repo** due to size > 25MB).
- `Bone_Fracture_Classifier using`: **PyTorch** and **TensorFlow** (not included here, but referenced in evaluation).
---
## 🔍 Problem Definition
Classify X-ray images into **10 different types of fractures** using a supervised image classification approach.
---
## 🧪 Methods & Approaches
Three models were developed and evaluated:
1. **Custom CNN using PyTorch**
2. **Custom CNN using TensorFlow**
3. **Transfer Learning using ResNet50 (Keras + TensorFlow)**

Each model used input images of size `128x128`, and trained on a multiclass fracture dataset.
---

## 🏗️ Model Architectures
### ✅ TensorFlow CNN
- 3 convolutional blocks (Conv2D → BatchNorm → LeakyReLU → Dropout)
- Fully Connected layer
- Output: 10-class softmax

### ✅ PyTorch CNN
- Similar architecture adapted to PyTorch

### ✅ ResNet50 (Transfer Learning)
- Pretrained on ImageNet
- Top layers replaced with:
  - GlobalAveragePooling
  - Dense → Dropout → Softmax

---
## 📊 Model Performance

| Model            | Train Acc | Val Acc | Test Acc |
|------------------|-----------|---------|----------|
| PyTorch CNN      | 58.26%    | 12.50%  | 17.18%   |
| TensorFlow CNN   | 96.33%    | 31.25%  | 29.78%   |
| **ResNet50**     | **91.80%**| **45.63%** | **41.57%** |

> 📈 ResNet50 significantly improved generalization and test accuracy.

---
## ✅ Key Learnings & Recommendations

- Transfer learning (ResNet50) outperformed custom CNNs in generalization.
- Using a larger and more diverse dataset can boost model performance.
- Augmentation techniques (contrast, zoom, rotation) can help capture complex patterns.
- Advanced hyperparamater tuning and model ensembling can further improve results.

---
## 🌐 Streamlit App (`app.py`) (requires resnet model to be save to same directory as app in .keras format)
This file contains a simple **Streamlit web app** to:
- Upload an X-ray image (`jpg`, `jpeg`, or `png`)
- Classify the image using a trained model
- Display prediction and confidence

