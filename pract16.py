# PCA is used for dimension reduction
# AIM:- to implement PCA (principal component analysis)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
data.keys()
print(data['target_names'])     #check the output classes
print(data['feature_names'])    #check the input attributes

df1 = pd.DataFrame(data['data'],columns=data['feature_names'])
scaling = StandardScaler()    #scale data before applying PCA
scaling.fit(df1)           #use fit and transform method
scaled_data= scaling.transform(df1)

principal = PCA(n_components=3)    #set the n_components=3
principal.fit(scaled_data)
x=principal.transform(scaled_data)

print(x.shape)    #check the dimensions of data after PCA

"Check the cvalues of eigen vectors produced"
"by principal components "
print(principal.components_)

#plot the components
plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],c=data['target'],cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()

#3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
axis = fig.add_subplot(111,projection='3d')      #choose projection= 3d for graph
#x[:,0]is pc1,x[:,1]is pc2 while x[:,2] is pc3
axis.scatter(x[:,0],x[:,2],c=data['target'],cmap='plasma')
axis.set_xlabel("PC1",fontsize=10)
axis.set_ylabel("PC2",fontsize=10)
axis.set_zlabel("PC3",fontsize=10)
plt.show()

#variance value explained by each of three principal components
print(principal.explained_variance_ratio_)