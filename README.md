# CNN Image Classification – Fashion MNIST & CIFAR-10

A set of Convolutional Neural Network (CNN) scripts built with
**TensorFlow / Keras** that classify images from the Fashion MNIST and
CIFAR-10 datasets. The project walks from a basic CNN to an improved,
better-regularized version, showing how to reduce overfitting and boost
accuracy.

## What's inside

**`5_cnn_fashion_mnist.py`**
A CNN that classifies **Fashion MNIST** images (clothing items like shirts,
shoes, and bags) into 10 categories — a step up from plain MNIST digits.

**`6_cnn_cifar.py`**
A basic CNN for the **CIFAR-10** dataset (10 classes: airplane, automobile,
bird, cat, deer, dog, frog, horse, ship, truck). It uses three strided
convolutional layers followed by dense layers, and reaches around **77%**
accuracy. The script shows that the model overfits, with training accuracy
climbing while validation accuracy flattens out.

**`7_cnn_cifar_improved.py`**
An improved CIFAR-10 CNN that fixes the overfitting problem using:
- A deeper **VGG-style** architecture (stacked Conv blocks)
- **Batch Normalization** after each conv layer
- **MaxPooling** instead of strided convolutions
- **Data augmentation** (random shifts and horizontal flips)
- More training epochs

The result is a better learning curve, higher validation accuracy, and
fewer misclassifications.

## Each script also includes

- Loading and normalizing the image data
- Building the model with the Keras **functional API**
- Training and plotting loss & accuracy curves
- A **confusion matrix** to see where the model gets confused
- Visualizing misclassified images with their true vs. predicted labels

## Tech stack

- **Python**
- **TensorFlow / Keras** – building and training the CNNs
- **numpy** – data handling
- **matplotlib** – plotting curves and images
- **scikit-learn** – confusion matrix

## Datasets

Both datasets are built into Keras and download automatically — no manual
setup needed:
- **Fashion MNIST** – 28×28 grayscale clothing images, 10 classes
- **CIFAR-10** – 32×32 color images, 10 classes

## Setup

1. Clone the repo:
```bash
   git clone https://github.com/aasari/CNN_CIFAR.git
   cd CNN_CIFAR
```

2. Install the dependencies:
```bash
   pip install tensorflow numpy matplotlib scikit-learn
```

## How to run

Run any of the scripts directly:

```bash
python 5_cnn_fashion_mnist.py
python 6_cnn_cifar.py
python 7_cnn_cifar_improved.py
```

The plots (loss curves, accuracy curves, confusion matrix, sample images)
display best in a Jupyter/Colab notebook, since these were originally
Colab notebooks.

## Note

Training the improved model (50 epochs with data augmentation) is much
faster on a **GPU** — Google Colab with a GPU runtime is a good free option.

## Author

**Aasari Prashanth** – Data Scientist
