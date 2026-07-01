import numpy as np
from utils import one_hot

def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_deriv(Z):
    return Z > 0


def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


def forward_prop(X, W1, b1, W2, b2):
    Z1 = W1.dot(X.T) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y.T
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)

        dW1, db1, dW2, db2 = backward_prop(
            Z1, A1, A2, W2, X, Y
        )

        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2,
            dW1, db1, dW2, db2,
            alpha
        )

        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy = {accuracy:.4f}")

    return W1, b1, W2, b2

def save_model(W1, b1, W2, b2):
    np.save("W1.npy", W1)
    np.save("b1.npy", b1)
    np.save("W2.npy", W2)
    np.save("b2.npy", b2)


def load_model():
    W1 = np.load("W1.npy")
    b1 = np.load("b1.npy")
    W2 = np.load("W2.npy")
    b2 = np.load("b2.npy")

    return W1, b1, W2, b2

def make_prediction(image, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(
        image.reshape(1, -1),
        W1,
        b1,
        W2,
        b2
    )

    prediction = get_predictions(A2)

    return prediction[0]