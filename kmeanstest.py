from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
data, _ = make_blobs(n_samples=1000, centers=4, random_state=42)

# Define number of clusters
k = 4

# Run k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42).fit(data)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data points and centroids
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.show()