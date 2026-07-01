# 🧠 MNIST Neural Network From Scratch (NumPy)

A feedforward neural network implemented entirely from scratch using **NumPy** to classify handwritten digits from the MNIST dataset. This project demonstrates the core mathematics behind neural networks without using deep learning frameworks such as TensorFlow, PyTorch, or Keras.

## Features

- Fully connected neural network built with NumPy
- Forward propagation
- Backpropagation
- Gradient descent optimization
- ReLU activation function
- Softmax output layer
- One-hot encoded labels
- Data normalization
- Training and validation split
- Model saving and loading
- Handwritten digit prediction and visualization using Matplotlib

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib

## Dataset

This project uses the **Kaggle Digit Recognizer** dataset based on the MNIST handwritten digit dataset.

- 42,000 labeled training images
- 28×28 grayscale images
- 784 input features
- 10 output classes (digits 0–9)

## Project Structure

```
MNIST-Neural-Network/
│
├── data/
│   └── train.csv
├── main.py
├── network.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Neural Network Architecture

```
Input Layer (784)
        │
        ▼
Hidden Layer (ReLU)
        │
        ▼
Output Layer (Softmax)
        │
        ▼
Predicted Digit
```

## Results

The network successfully learns to classify handwritten digits using gradient descent and backpropagation.

Implemented features include:

- Forward propagation
- Backpropagation
- Weight updates using gradient descent
- Accuracy evaluation
- Validation testing
- Visualization of model predictions

## Future Improvements

- Increase hidden layer size
- Multiple hidden layers
- Mini-batch gradient descent
- Adam optimizer
- Convolutional Neural Networks (CNNs)
- Interactive digit drawing interface

## What I Learned

This project strengthened my understanding of:

- Neural network architecture
- Matrix operations with NumPy
- Forward and backward propagation
- Gradient descent optimization
- Data preprocessing
- Scientific computing in Python

## License

This project is intended for educational purposes.
