import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, normalize
from network import (
    gradient_descent,
    forward_prop,
    get_predictions,
    get_accuracy,
    make_prediction,
    save_model
)

# ==========================================
# Load and prepare the dataset
# ==========================================

data = load_data("data/train.csv").to_numpy()

# Shuffle the data
np.random.shuffle(data)

# Split into validation and training sets
validation = data[:1000]
training = data[1000:]

# Validation set
Y_val = validation[:, 0]
X_val = normalize(validation[:, 1:])

# Training set
Y_train = training[:, 0]
X_train = normalize(training[:, 1:])

# ==========================================
# Train the neural network
# ==========================================

W1, b1, W2, b2 = gradient_descent(
    X_train,
    Y_train,
    alpha=0.1,
    iterations=100
)

# ==========================================
# Save trained model
# ==========================================

save_model(W1, b1, W2, b2)

print("\n✅ Model saved successfully!")

# ==========================================
# Evaluate on validation set
# ==========================================

_, _, _, A2 = forward_prop(X_val, W1, b1, W2, b2)

predictions = get_predictions(A2)

validation_accuracy = get_accuracy(predictions, Y_val)

print(f"Validation Accuracy: {validation_accuracy:.4f}")

# ==========================================
# Display 9 predictions
# ==========================================

print("Opening prediction window...")

fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for i, ax in enumerate(axes.flat):

    prediction = make_prediction(
        X_val[i],
        W1,
        b1,
        W2,
        b2
    )

    ax.imshow(X_val[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {prediction}\nActual: {Y_val[i]}")
    ax.axis("off")

plt.tight_layout()

plt.show(block=True)

print("Image window closed.")