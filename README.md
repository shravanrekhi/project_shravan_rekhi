# 🫁 Pneumonia Detection via Chest X-Rays

## 📋 Project Overview
This project implements a custom Deep Learning Convolutional Neural Network (CNN) in PyTorch to detect signs of Pneumonia from chest X-ray images. The model was trained using the `ChestMNIST` dataset and utilizes specialized data augmentation, Batch Normalization, and a weighted loss function to handle class imbalances effectively.

## 📁 Repository Structure
This repository adheres strictly to the required grading format. All executable files and data are located at the root level.
```text
├── _checkpoints/
│   └── _final_weights.pth      # Pre-trained model weights
├── data/                       # 20 raw .jpg test images (10 per class)
│   ├── normal_0.jpg
│   ├── normal_1.jpg
│   ├── normal_2.jpg
│   ├── normal_3.jpg
│   ├── normal_4.jpg
│   ├── normal_5.jpg
│   ├── normal_6.jpg
│   ├── normal_7.jpg
│   ├── normal_8.jpg
│   ├── normal_9.jpg
│   ├── pneumonia_0.jpg
│   ├── pneumonia_1.jpg
│   ├── pneumonia_2.jpg
│   ├── pneumonia_3.jpg
│   ├── pneumonia_4.jpg
│   ├── pneumonia_5.jpg
│   ├── pneumonia_6.jpg
│   ├── pneumonia_7.jpg
│   ├── pneumonia_8.jpg
│   └── pneumonia_9.jpg
├── config.py                   # Hyperparameters and dataset constants
├── dataset.py                  # Custom PyTorch Dataset and DataLoader logic
├── interface.py                # Standardized import file for the grading script
├── model.py                    # PyTorch Sequential CNN architecture definition
├── predict.py                  # Inference function for evaluating new batches
├── train.py                    # Model training loop
└── README.md
