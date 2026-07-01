# MNIST Neural Network From Scratch (NumPy)

A handwritten digit recognition neural network implemented entirely from scratch using **NumPy**. This project demonstrates the core mathematics behind neural networks without using machine learning frameworks such as TensorFlow or PyTorch.

---

## Overview

This project implements a complete feedforward neural network capable of recognizing handwritten digits from the MNIST dataset.

The model is built from first principles, including:

- Forward propagation
- ReLU activation
- Softmax output layer
- Backpropagation
- Gradient descent
- Model evaluation
- Model saving/loading
- Validation set testing
- Image prediction and visualization

The objective is to understand how neural networks work internally rather than relying on high-level machine learning libraries.

---

## Features

- Neural network implemented entirely with NumPy
- No TensorFlow, PyTorch or Keras
- Modular project structure
- Data preprocessing
- One-hot encoding
- ReLU activation
- Softmax classifier
- Backpropagation
- Gradient descent optimization
- Training accuracy tracking
- Validation dataset evaluation
- Save and load trained parameters
- Visualize handwritten digit predictions

---

## Dataset

This project uses the Kaggle **Digit Recognizer** dataset, which is based on the MNIST handwritten digit dataset.

Each image consists of:

- 28 × 28 grayscale pixels
- 784 input features
- Labels from 0–9

---

## Project Structure

```
MNIST-Neural-Network/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── network.py
├── utils.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── W1.npy
├── W2.npy
├── b1.npy
└── b2.npy
```

---

## Neural Network Architecture

```
784 Input Neurons
        │
        ▼
Fully Connected Hidden Layer
        │
      ReLU
        │
        ▼
10 Output Neurons
        │
     Softmax
        │
        ▼
Predicted Digit
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/MNIST-Neural-Network.git
```

Navigate to the project:

```bash
cd MNIST-Neural-Network
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Train the neural network:

```bash
python main.py
```

The program will:

- Load the MNIST dataset
- Train the neural network
- Evaluate performance on the validation dataset
- Save the trained model
- Display handwritten digit predictions

---

## Example Output

```
Iteration 0: Accuracy = 0.10
Iteration 10: Accuracy = 0.34
Iteration 20: Accuracy = 0.56
Iteration 30: Accuracy = 0.71
Iteration 40: Accuracy = 0.80
Iteration 50: Accuracy = 0.86
Iteration 60: Accuracy = 0.89
Iteration 70: Accuracy = 0.91
Iteration 80: Accuracy = 0.92
Iteration 90: Accuracy = 0.93

Model saved successfully!

Validation Accuracy: 0.91
```

---

## How It Works

1. Load and normalize the MNIST dataset.
2. Initialize random weights and biases.
3. Perform forward propagation.
4. Calculate prediction error.
5. Compute gradients using backpropagation.
6. Update weights using gradient descent.
7. Repeat until the model converges.
8. Evaluate on unseen validation data.

---

## Future Improvements

Potential improvements include:

- Multiple hidden layers
- Mini-batch gradient descent
- Adam optimizer
- Dropout regularization
- Convolutional Neural Networks (CNNs)
- Interactive digit drawing interface
- Hyperparameter tuning
- GPU acceleration

---

## Learning Outcomes

This project strengthened my understanding of:

- Linear algebra in machine learning
- Matrix operations with NumPy
- Neural network architecture
- Forward and backward propagation
- Gradient descent optimization
- Model evaluation
- Scientific computing in Python

---

## License

This project is intended for educational purposes.