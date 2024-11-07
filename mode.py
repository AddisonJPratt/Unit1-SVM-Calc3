# %%
# Import necessary libraries
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import json

# %%
# Define image dimensions
IMAGE_SIZE = (256, 256)  # Increased image size for better feature representation

# Paths to your image directories
cat_images_path = "/Users/addisonjpratt/Documents/Repositories/Project Repositories/Calc-3-Projects/Unit-1 SVM/data/train/cat/"
dog_images_path = "/Users/addisonjpratt/Documents/Repositories/Project Repositories/Calc-3-Projects/Unit-1 SVM/data/train/dog"


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        # Ensure that the file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")  # Convert to RGB
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            labels.append(label)
    return images, labels


# Load cat images (label +1)
cat_images, cat_labels = load_images_from_folder(cat_images_path, 1)

# Load dog images (label -1)
dog_images, dog_labels = load_images_from_folder(dog_images_path, -1)

# Combine data
X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)


# Data augmentation: Horizontal flip and rotation
def augment_images(images, labels):
    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        # Horizontal flip
        flipped_img = np.fliplr(img)
        augmented_images.append(flipped_img)
        augmented_labels.append(label)
        # Rotation
        rotated_img = np.rot90(img)
        augmented_images.append(rotated_img)
        augmented_labels.append(label)
    return augmented_images, augmented_labels


aug_images, aug_labels = augment_images(X, y)
X = np.concatenate((X, aug_images))
y = np.concatenate((y, aug_labels))


# Extract features using pre-trained CNN (VGG16)
def extract_cnn_features(images):
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    # Preprocess images
    images_preprocessed = preprocess_input(images * 255.0)
    features = base_model.predict(images_preprocessed, batch_size=32, verbose=1)
    features_flattened = features.reshape(features.shape[0], -1)
    return features_flattened


# Extract features
X_features = extract_cnn_features(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

import joblib  # Import joblib to save and load models

# Check if the scaler model is already saved
if os.path.exists("scaler.save"):
    # Load the saved scaler
    scaler = joblib.load("scaler.save")
    print("Scaler loaded from disk.")
    # Transform features using the loaded scaler
    X_scaled = scaler.transform(X_features)
else:
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    # Save the scaler to disk
    joblib.dump(scaler, "scaler.save")
    print("Scaler trained and saved to disk.")

# Dimensionality reduction with PCA
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

joblib.dump(pca, "pca.save")
print("PCA model trained and saved to disk.")

# Shuffle data
indices = np.random.permutation(len(X_pca))
X_pca = X_pca[indices]
y = y[indices]

# Split into training and testing sets
split_index = int(0.8 * len(X_pca))
X_train, X_test = X_pca[:split_index], X_pca[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# Define the manually implemented SVM class
class SVM:
    def __init__(self, learning_rate=0.0001, lambda_param=0.001, n_iters=5000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        for iteration in range(self.n_iters):
            # Shuffle the data to ensure randomness
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            # Initialize gradients
            dw = np.zeros(n_features)
            db = 0
            loss = 0

            for idx in range(n_samples):
                xi = X[idx]
                yi = y[idx]
                # Compute the decision function
                condition = yi * (np.dot(xi, self.w) + self.b)
                if condition < 1:
                    # Misclassified or within margin
                    dw += -yi * xi
                    db += -yi
                    loss += 1 - condition
                # Regularization term
            # Average gradients
            dw = dw / n_samples + self.lambda_param * self.w
            db = db / n_samples

            # Update weights and bias
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Compute loss
            loss = (0.5 * self.lambda_param * np.dot(self.w, self.w)) + loss / n_samples

            # Record the loss
            self.loss_history.append(loss)

            # Optionally, print progress
            if (iteration + 1) % 500 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iters}, Loss: {loss}")

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, -1)

    def save_weights(self, filename="svm_weights.json"):
        # Save the weights and bias to a JSON file
        model_params = {"weights": self.w.tolist(), "bias": self.b}
        with open(filename, "w") as file:
            json.dump(model_params, file)
        print(f"Weights and bias saved to {filename}")

    def load_weights(self, filename="Unit-1 SVM/svm_weights.json"):
        # Load the weights and bias from a JSON file
        with open(filename, "r") as file:
            model_params = json.load(file)
        self.w = np.array(model_params["weights"])
        self.b = model_params["bias"]
        print(f"Weights and bias loaded from {filename}")


# Hyperparameters
learning_rate = 0.0001  # Adjusted learning rate
lambda_param = 0.001  # Adjusted regularization parameter
n_iters = 5000  # Increased iterations for better convergence

# Initialize and train the SVM
svm = SVM(learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
svm.fit(X_train, y_train)
svm.save_weights("svm_weights.json")

# Plot the loss over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(len(svm.loss_history)), svm.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss over iterations")
plt.show()

# Make predictions
predictions = svm.predict(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"SVM classification accuracy: {accuracy * 100:.2f}%")

# %%

# %%
