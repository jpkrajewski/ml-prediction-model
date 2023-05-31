import numpy as np
from sklearn.decomposition import PCA

# Create a sample dataset
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Create a PCA instance
pca = PCA(n_components=2)  # Specify the number of components to keep

# Fit the PCA model to the data
pca.fit(X)

# Transform the data to the lower-dimensional representation
X_pca = pca.transform(X)

# Access the principal components (eigenvectors)
components = pca.components_

# Access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Print the transformed data and explained variance ratio
print("Transformed Data:")
print(X_pca)
print("Explained Variance Ratio:")
print(explained_variance_ratio)