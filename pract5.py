# AIM:1B   UNIVARIATE ,BIAVARIATE , MULTIVARIATE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\91892\Downloads\Iris.csv")

#univarite analysis
df_setosa = df.loc[df["Species"]=="Iris-setosa"]
df_virginica = df.loc[df["Species"]=="Iris-virginica"]
df_versicolor = df.loc[df["Species"]=="Iris-versicolor"]

plt.plot(df_setosa["Sepal_length"],np.zeros_like(df_setosa["Sepal_length"]),'o')
plt.plot(df_virginica["Sepal_length"],np.zeros_like(df_virginica["Sepal_length"]),'o')
plt.plot(df_versicolor["Sepal_length"],np.zeros_like(df_versicolor["Sepal_length"]),'o')
plt.xlabel("Sepal_length")
plt.show()

#bivariate analysis
sns.FacetGrid(df,hue="Species").map(plt.scatter,"Petal_length","Sepal_width").add_legend();
plt.show()

#multivariate
sns.pairplot(df,hue="Species")
plt.show()