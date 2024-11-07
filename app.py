import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib
import json
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define image dimensions
IMAGE_SIZE = (264, 264)

# Load the saved scaler, PCA models, and SVM weights
scaler = joblib.load("scaler.save")
pca = joblib.load("pca.save")


class SVM:
    def __init__(self):
        self.w = None
        self.b = None

    def load_weights(self, filename="svm_weights.json"):
        with open(filename, "r") as file:
            model_params = json.load(file)
        self.w = np.array(model_params["weights"])
        self.b = model_params["bias"]

    def predict(self, X):
        decision_value = np.dot(X, self.w) + self.b
        prediction = np.where(decision_value >= 0, 1, -1)
        return prediction, decision_value


# Function to extract features from the image using VGG16
def extract_features(img):
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    images_preprocessed = preprocess_input(img_array * 255.0)
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    features = base_model.predict(images_preprocessed)
    return features.reshape(features.shape[0], -1)


def plot_decision_boundary(features, labels, svm_model, pca_2d):
    # Reduce original features to 2D for visualization
    features_2d = pca_2d.transform(features)
    
    # Generate grid for decision boundary
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict decision values on the 2D grid using SVM weights directly in 2D
    decision_values = np.dot(grid, svm_model.w[:2]) + svm_model.b  # Use only the first 2 components
    zz = decision_values.reshape(xx.shape)
    
    # Plot decision boundary and data points
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, zz, levels=[zz.min(), 0, zz.max()], alpha=0.2, colors=["#FFAAAA", "#AAAAFF"])
    plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors="k")  # Hyperplane

    # Plot data points in 2D space
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="bwr", edgecolor="k", s=50)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("SVM Decision Boundary with Data Points")
    plt.legend(handles=scatter.legend_elements()[0], labels=['Dog', 'Cat'])
    st.pyplot(plt.gcf())  # Display plot in Streamlit

# Main application
def main():
    st.title("Cat vs Dog Classifier with SVM Visualization")
    st.write(
        "Upload an image of a cat or a dog, and the classifier will predict which one it is."
    )

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image.", use_container_width=True)
        st.write("")

        # Step 1: Feature extraction
        st.subheader("Step 1: Feature Extraction with VGG16")
        features = extract_features(image)
        st.write(
            "The image is resized to 256x256 pixels and passed through the VGG16 network to extract features."
        )
        st.write("Feature shape:", features.shape)

        # Step 2: Standardize features using the loaded scaler
        st.subheader("Step 2: Standardization")
        features_scaled = scaler.transform(features)

        # Step 3: PCA transformation to reduce dimensionality
        st.subheader("Step 3: Dimensionality Reduction with PCA")
        features_pca = pca.transform(features_scaled)

        # Load and apply SVM model
        st.subheader("Step 4: Classification with SVM")
        svm = SVM()
        svm.load_weights("svm_weights.json")

        # Make prediction and calculate decision value
        prediction, decision_value = svm.predict(features_pca)
        label_map = {1: "Cat", -1: "Dog"}
        result = label_map.get(prediction[0], "Unknown")

        # Display prediction result
        st.write(f"**Prediction: {result}**")

        # Calculate and display confidence based on distance from hyperplane
        confidence_score = 1 / (1 + np.exp(-decision_value[0]))
        st.write(f"Confidence Score: {confidence_score:.2f}")

        # Explanation of SVM classification
        st.subheader("How SVM Makes Its Prediction")
        st.write(
            """
            SVM finds a hyperplane that separates the classes in the feature space. 
            Points on one side are classified as 'Cat' and points on the other as 'Dog'.
            The distance of each point from the hyperplane indicates the confidence in the classification.
        """
        )

        # Visualize the decision boundary and data points
        if st.checkbox("Show Decision Boundary and Data Points"):
            st.subheader("Visualization of SVM Decision Boundary")

            # Simulate additional data points for visualization (or replace with your dataset points)
            num_samples = 20
            cat_points = np.random.normal(
                loc=0, scale=1, size=(num_samples, features_pca.shape[1])
            )
            dog_points = np.random.normal(
                loc=1, scale=1, size=(num_samples, features_pca.shape[1])
            )
            all_features = np.vstack((cat_points, dog_points))
            all_labels = np.array([1] * num_samples + [-1] * num_samples)

            # Reduce dimensionality to 2D for visualization
            pca_2d = PCA(n_components=2)
            all_features_2d = pca_2d.fit_transform(all_features)

            # Plot
            plot_decision_boundary(all_features, all_labels, svm, pca_2d)


if __name__ == "__main__":
    main()
