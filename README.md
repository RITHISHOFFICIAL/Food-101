Here is a comprehensive, production-ready `README.md` file tailored to your Kaggle project and problem statement.

---

# Food-101 Image Classification using Custom Residual CNN

This repository contains a PyTorch-based Deep Learning pipeline designed to classify food images into 101 distinct categories using the **Food-101** dataset. The model architecture uses custom residual blocks to efficiently learn deep spatial hierarchies while preventing the vanishing gradient problem.

---

## Project Overview

* **Framework:** PyTorch
* **Architecture:** Deep Convolutional Neural Network with Residual Connections (ResNet-style)
* **Dataset:** Food-101 (`food41` Kaggle variant)
* **Task:** Multi-class image classification (101 classes)
* **Training Duration:** 25 Epochs
* **Final Training Accuracy:** **95.87%**

---

## Dataset Structure

The notebook expects the dataset to be structured according to standard Kaggle directory conventions for `food41`:

* `/kaggle/input/food41/images`: Directory containing subfolders for each of the 101 classes.
* `/kaggle/input/food41/meta/meta/`: Houses split metadata (`train.txt` and `test.txt`).

### Data Preprocessing & Augmentation

Images are dynamically loaded using `PIL` and passed through a transformation pipeline:

* **Resizing:** Scaled to $128 \times 128$ pixels.
* **Normalization:** Standardized using ImageNet parameters:
* Mean: `[0.485, 0.456, 0.406]`
* Standard Deviation: `[0.229, 0.224, 0.225]`



---

##  Model Architecture

The network design leverages a custom block structure to improve optimization performance over 25 epochs.

### 1. Residual Block (`ResidualBlock`)

Consists of two consecutive 3x3 Conv2d layers, Batch Normalization, and ReLU activations. If a dimension mismatch occurs during downsampling, a 1x1 Conv2d layer dynamically scales the shortcut projection to perform an identity skip connection:


$$\text{Output} = \text{ReLU}(\text{Layer}_2(\text{ReLU}(\text{Layer}_1(x))) + \text{Shortcut}(x))$$

### 2. Global Network Wrapper (`CNN`)

* **Preparation Layer:** 7x7 Convolution (stride 2, padding 3) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ 3x3 MaxPool.
* **Layer 1:** 2 $\times$ Residual Blocks (64 channels)
* **Layer 2:** 2 $\times$ Residual Blocks (128 channels, downsampling)
* **Layer 3:** 2 $\times$ Residual Blocks (256 channels, downsampling)
* **Layer 4:** 2 $\times$ Residual Blocks (512 channels, downsampling)
* **Classifier Head:** Adaptive Average Pooling $\rightarrow$ Flatten $\rightarrow$ Dropout (0.5) $\rightarrow$ Fully Connected Layer (101 Outputs).

---

##  Training Hyperparameters

* **Loss Function:** Cross Entropy Loss (`nn.CrossEntropyLoss`)
* **Optimizer:** Adam
* **Learning Rate:** 1e-4
* **Batch Size:** 64
* **Workers:** 8 (`pin_memory=True`, `persistent_workers=True` optimized for CUDA runtime)

---

##  Results & Visualization

### Training Performance Matrix

The model displays stable convergence behavior across the 25-epoch regimen:

* **Epoch 1:** Loss = 4.0834, Accuracy = 9.19%
* **Epoch 10:** Loss = 0.7592, Accuracy = 79.07%
* **Epoch 25:** Loss = 0.1332, Accuracy = **95.87%**

### Evaluation Metrics Included

1. **Convergence Plots:** The notebook renders live line graphs modeling the steady optimization decay of **Training Loss** alongside the step-wise acceleration of **Training Accuracy**.
2. **Confusion Matrix:** Renders an evaluated validation slice mapping true versus predicted labels for the first 10 food classes using `sklearn.metrics.ConfusionMatrixDisplay`.

---

##  Model Serialization

Upon completion of the final epoch, model weights are packaged and serialized directly inside your root environment directory using `pickle`:

```python
with open("cnn_model_weights.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)

```

To reload these weights for inference downstream:

```python
model = CNN()
with open("cnn_model_weights.pkl", "rb") as f:
    model.load_state_dict(pickle.load(f))
model.eval()

```

---

## 🛠️ Required Dependencies

Ensure the following packages are configured inside your working notebook kernel:

* `torch`
* `torchvision`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `pillow`
