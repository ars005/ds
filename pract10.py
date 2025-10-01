#AIM: 5.1 To implement clustering using K-Means Algorithm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Prepare data
data = {
    'Age': [22, 23, 25, 24, 26,      #grp 1: young, loe salary, short browsing
            35, 36, 34, 33, 37,      #grp 2: mid-age
            48, 50, 52, 49, 51,      #grp 3: 
            23, 36, 50, 35, 48],
    'Salary': [25000, 27000, 26000, 28000, 24000,
               60000, 62000, 58000, 61000, 59000,
               100000, 98000, 105000, 97000, 102000,
               25500, 60500, 101000, 61500, 99000],
    'Browsing_Time': [1.5, 1.8, 2.0, 1.6, 1.9,
                      5.0, 5.2, 4.8, 5.5, 5.1,
                      9.0, 8.5, 9.2, 8.8, 9.5,
                      2.0, 5.3, 9.0, 5.0, 8.7]
}
df = pd.DataFrame(data)

# Step 2: Scale the data/ scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(df[['Age', 'Salary', 'Browsing_Time']])

# Step 3: KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)   #output will give 3 group bcoz 3 cluster
df['Cluster'] = kmeans.fit_predict(scaled)   #scaled is holding the data (age n all)

# Step 4: Plotting / plot clusters(age vs salary)
plt.figure(figsize=(8, 6))
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Age'], cluster_data['Salary'], label=f'Cluster {cluster}', s=100)

# Plot formatting (outside loop)
plt.title('Clustered Data (Age vs Salary)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
