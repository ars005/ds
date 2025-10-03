# Clustering Algorithm
# 5.1) To implement clustering using K-Means Algorithm
# 5.2) To perform hierarchical clustering

# a. k-means
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Synthetic cluster-friendly dataset
data = {
    "Age": [
        22,
        23,
        25,
        24,
        26,  # Group 1: Young, low salary, short browsing
        35,
        36,
        34,
        33,
        37,  # Group 2: Mid-age, mid salary, medium browsing
        48,
        50,
        52,
        49,
        51,  # Group 3: Older, high salary, long browsing
        23,
        36,
        50,
        35,
        48,
    ],  # Mix for variation
    "Salary": [
        25000,
        27000,
        26000,
        28000,
        24000,
        60000,
        62000,
        58000,
        61000,
        59000,
        100000,
        98000,
        105000,
        97000,
        102000,
        25500,
        60500,
        101000,
        61500,
        99000,
    ],
    "Browsing_Time": [
        1.5,
        1.8,
        2.0,
        1.6,
        1.9,
        5.0,
        5.2,
        4.8,
        5.5,
        5.1,
        9.0,
        8.5,
        9.2,
        8.8,
        9.5,
        2.0,
        5.3,
        9.0,
        5.0,
        8.7,
    ],
}

df = pd.DataFrame(data)

# Step 2: Scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(df[["Age", "Salary", "Browsing_Time"]])

# Step 3: KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled)

# Step 4: Plot clusters (Age vs Salary)
plt.figure(figsize=(8, 6))
for cluster in df["Cluster"].unique():
    cluster_data = df[df["Cluster"] == cluster]
    plt.scatter(
        cluster_data["Age"],
        cluster_data["Browsing_Time"],
        label=f"Cluster {cluster}",
        s=100,
    )

plt.title("Clustered Data (Age vs Salary)")
plt.xlabel("Age")
plt.ylabel("Browsing_Time")
plt.legend()
plt.grid(True)
plt.show()

# b.  Hierarchical Clustering Agglomerative

# AIM: 5.2 To perform hierarchical clustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# step 1 : prepare the dataset
data = {
    "Age": [
        22,
        25,
        47,
        52,
        46,
        56,
        48,
        55,
        60,
        32,
        40,
        28,
        38,
        29,
        30,
        41,
        26,
        34,
        45,
        50,
    ],
    "Salary": [
        25000,
        27000,
        90000,
        110000,
        95000,
        120000,
        99000,
        105000,
        115000,
        48000,
        80000,
        30000,
        75000,
        32000,
        35000,
        82000,
        28000,
        60000,
        87000,
        100000,
    ],
    "Browsing_Time": [
        1.5,
        2.0,
        8.5,
        9.0,
        7.5,
        10.0,
        7.0,
        8.0,
        9.5,
        3.5,
        6.5,
        2.5,
        6.0,
        3.0,
        3.2,
        7.0,
        2.2,
        4.5,
        6.8,
        8.5,
    ],
}
df = pd.DataFrame(data)

# step 2 : normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# step 3: dendrogram for visualization
plt.figure(figsize=(10, 6))
linked = linkage(X_scaled, method="ward")
dendrogram(linked, orientation="top", distance_sort="ascending", show_leaf_counts=True)
plt.title("Dendrogram - heirarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Step:4 apply agglomarative clustering (e.g , 3 clusters)
cluster = AgglomerativeClustering(n_clusters=3, linkage="ward")
df["Cluster"] = cluster.fit_predict(X_scaled)

# Step: 5 show result
print("\nClustered Data:")
print(df[["Age", "Salary", "Browsing_Time", "Cluster"]])

# optional:visualize clusters
sns.scatterplot(data=df, x="Salary", y="Browsing_Time", hue="Cluster", palette="deep")
plt.title("hierarchical Clustering Result")
plt.xlabel("Salary")
plt.ylabel("Browsing Time")
plt.grid(True)
plt.show()
