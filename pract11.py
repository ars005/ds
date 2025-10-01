#AIM: 5.2 To perform hierarchical clustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

#step 1 : prepare the dataset
data = {
    'Age':[22, 25, 47, 52, 46, 56, 48, 55, 60, 32,
           40, 28, 38, 29, 30, 41, 26, 34, 45, 50],
    'Salary':[25000, 27000, 90000, 110000, 95000, 120000, 99000, 105000, 115000, 48000,
              80000, 30000, 75000, 32000, 35000, 82000, 28000, 60000, 87000, 100000],
    'Browsing_Time':[1.5, 2.0, 8.5, 9.0, 7.5, 10.0, 7.0, 8.0, 9.5, 3.5,
                     6.5, 2.5, 6.0, 3.0, 3.2, 7.0, 2.2, 4.5, 6.8, 8.5]
}
df = pd.DataFrame(data)

#step 2 : normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#step 3: dendrogram for visualization
plt.figure(figsize=(10,6))
linked = linkage(X_scaled, method='ward')
dendrogram(linked,
           orientation='top',
           distance_sort='ascending',
           show_leaf_counts=True
           )
plt.title("Dendrogram - heirarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

#Step:4 apply agglomarative clustering (e.g , 3 clusters)
cluster= AgglomerativeClustering(n_clusters=3, linkage='ward')
df['Cluster'] = cluster.fit_predict(X_scaled)

#Step: 5 show result
print("\nClustered Data:")
print(df[['Age','Salary','Browsing_Time','Cluster']])

#optional:visualize clusters
sns.scatterplot(data=df, x='Salary', y='Browsing_Time', hue='Cluster', palette='deep')
plt.title("hierarchical Clustering Result")
plt.xlabel("Salary")
plt.ylabel("Browsing Time")
plt.grid(True)
plt.show()