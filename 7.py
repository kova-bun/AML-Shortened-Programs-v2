import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)
x = np.random.rand(100,2)
m = KMeans(n_clusters=3,random_state=42).fit(x)

plt.scatter(x[:,0],x[:,1],c=m.labels_,cmap="viridis")
plt.scatter(*m.cluster_centers_.T,c="red",marker="X",s=200)
plt.title("K-Means Clustering")
plt.show()
