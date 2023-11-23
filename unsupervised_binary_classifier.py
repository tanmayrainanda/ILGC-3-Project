import os
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

# Load the dataset
image_dir = '/Users/tanmay/Documents/GitHub/ILGC-3-Project/Collected Dataset'

# Preprocess the images and extract temperature values
def preprocess_images(image_dir):
    images = []
    temperatures = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            temperature = np.mean(img)  # Calculate the average temperature value
            images.append(img)
            temperatures.append(temperature)
    return np.array(images), np.array(temperatures)

# Load and preprocess the dataset
images, temperatures = preprocess_images(image_dir)

# Remove outliers using Isolation Forest
scaler = MinMaxScaler()
temperatures_standardized = scaler.fit_transform(temperatures.reshape(-1, 1))
outlier_detector = IsolationForest(contamination=0.05, random_state=0)
is_inlier = outlier_detector.fit_predict(temperatures_standardized)
images = images[is_inlier == 1]
temperatures = temperatures[is_inlier == 1]

# Experiment with the number of components (clusters) and covariance type
num_components = 4  # Hot, Cold, and Moderate

cluster_names = {
    0: "Very Hot", 
    1: "Hot",
    2: "Moderate",
    3: "Cold",
    4: "Very Cold"
}

# Experiment with covariance type
covariance_type = 'full'  # Try 'full' instead of 'diag'

# Create a GMM model
gmm_model = GaussianMixture(n_components=num_components, covariance_type=covariance_type, random_state=0)

# Fit the GMM model to the scaled temperature data
gmm_model.fit(temperatures.reshape(-1, 1))
image_clusters = gmm_model.predict(temperatures.reshape(-1, 1))

# Visualize the clustered images using PCA for dimensionality reduction
def plot_clusters(images, labels):
    tsne = TSNE(n_components=2, random_state=0,perplexity=50)
    reduced_data = tsne.fit_transform(images.reshape(images.shape[0], -1))
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
    plt.legend(handles=scatter.legend_elements()[0], labels=cluster_names.values())
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("Cluster Visualization")
    plt.show()

plot_clusters(images, image_clusters)

def classify_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        temperature = np.mean(img)
        cluster = gmm_model.predict([[temperature]])
        cluster_name = cluster_names[cluster[0]]
        return cluster_name

image_path = "/Users/tanmay/Documents/GitHub/ILGC-3-Project/Collected Dataset/FLIR00154.jpg"

# Print the identified clusters
for i in range(len(images)):
    print("Image: ", i, "Temperature: ", temperatures[i], "Cluster: ", cluster_names[image_clusters[i]])

print("Classified Temperature for the image:", classify_image(image_path))

# Calculate the silhouette score
silhouette_avg = silhouette_score(temperatures.reshape(-1, 1), image_clusters)
print("Silhouette Score: ", silhouette_avg)