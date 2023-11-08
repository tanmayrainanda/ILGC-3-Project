import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

# Load the dataset
image_dir = 'Data_img'

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

# Standardize the temperature values
scaler = StandardScaler()
temperatures_standardized = scaler.fit_transform(temperatures.reshape(-1, 1))

# Define the model for clustering
dbscan_model = DBSCAN(eps=0.5, min_samples=5)

# Train the model
dbscan_model.fit(temperatures_standardized)

# Predict the clusters
labels = dbscan_model.labels_

# Define the cluster names
cluster_names = {
    -1: "Outlier",
    0: "Hot",
    1: "Moderate",
    2: "Cold"
}

def classify_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        temperature = np.mean(img)
        temperature_standardized = scaler.transform([[temperature]])
        cluster = dbscan_model.fit_predict(temperature_standardized)
        cluster_name = cluster_names[cluster[0]]
        return cluster_name

image_path = "/Users/tanmay/Documents/GitHub/ILGC-3-Project/Collected Dataset/FLIR0054.jpg"


#print all identified clusters
for i in range(len(images)):
    print("Image: ", i, "Cluster: ", cluster_names[labels[i]])

# Calculate the silhouette score
silhouette_avg = silhouette_score(temperatures_standardized, labels)
print("Silhouette Score: ", silhouette_avg)

print(classify_image(image_path))

#print the accuracy of the model
print("Accuracy: ", np.sum(labels == 0) / len(labels))


cv2.imshow('image', images[8])
cv2.waitKey(0)
cv2.destroyAllWindows()