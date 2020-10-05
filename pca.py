"""
Created on Mon Oct  5 13:16:19 2020

SHORT PRINCIPAL COMPONENT ANALYSIS PROJECT

@author: Marc
"""
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Loading the data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
data = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])

# Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data) # shape (569, 30)

# Building the PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data) # shape (569, 2)

# Visualizing the PCA components based on target column of the data
plt.figure(figsize = (9, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = cancer['target'], cmap = 'plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('2 Principal Components (2 dimensions)')
plt.savefig('components_scatter_plot.png')

# Visualizing the correlation between PCA components and data features
df_comp = pd.DataFrame(pca.components_, columns = cancer['feature_names'])
plt.figure(figsize = (9, 6))
sns.heatmap(df_comp, cmap = 'plasma')
plt.tight_layout()
plt.title('Correlation between Components and Features')
plt.savefig('correlation_heatmap.png')