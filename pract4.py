#AIM : To implement data cleaning To detect outliers in the given data.
import pandas as pd
import numpy as np

InputFileName='Movie_collection_train.csv'

print('###################')
print("Input file")
sFileName=r"C:\Users\91892\Downloads\Movie_collection_train.csv"
print('Loading :',sFileName)
Movie_DATA_ALL = pd.read_csv(sFileName, header=0, usecols=['Genre', '3D_available', 'Budget'], encoding='latin-1')
Movie_DATA_ALL.rename(columns={'Genre':'Movie type'},inplace=True)
print(Movie_DATA_ALL)
MeanData=Movie_DATA_ALL.groupby(['Movie type','3D_available'])['Budget'].mean()
stdData=Movie_DATA_ALL.groupby(['Movie type','3D_available'])['Budget'].std()
print(MeanData);
print(stdData);
print('Outliers')

UpperBound = float(sum(MeanData) + sum(stdData))
print('Higher than ', UpperBound)
OutliersHigher = Movie_DATA_ALL[Movie_DATA_ALL.Budget > UpperBound]
print(OutliersHigher)

LowerBound = float(sum(MeanData) - sum(stdData))
print('Lower than ', LowerBound)
OutliersLower = Movie_DATA_ALL[Movie_DATA_ALL.Budget < LowerBound]
print(OutliersLower)

print('Not Outliers')
OutliersNot = Movie_DATA_ALL[(Movie_DATA_ALL.Budget > LowerBound) & (Movie_DATA_ALL.Budget <= UpperBound)]
print(OutliersNot)
