import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import numpy as np

#Phase 1 : Data Exploration and Preprocessing

#1) DATA LOADING

#Load the data - Skip "Land Ocean Global Means
data  = pd.read_csv('Global_Temp.csv', skiprows = 1)

data = pd.DataFrame(data)

#Data Overview
print(data.head(10))

#Data Info
print("Data Info:\n")
print(data.info())

#As seen in the table below, column Year is the only Integer column, while
#D-N and DJF are the only object columns and the remaining ones are floats.
#The 2 will be handled later and will be converted to the right format (float).

#Statistical Summary - Overview of the dataset, gives good description for numeric dataset
#displayinh count, mean, std, min and max

print("Data Describe:\n")
print(data.describe())


#Missing Values - No Null/Nan values, although will require to impute 0 with K-NN imputation
print("Dataset Null")
print(data.isnull().sum())


#2) DATA CLEANING

#Convert Columns: D-N and DJF to numeric

data['D-N'] = pd.to_numeric(data['D-N'], errors='coerce')
data['DJF'] = pd.to_numeric(data['DJF'], errors='coerce')

#Impute 0 values using KNN imputation method
#Replace 0 with NaN
data.replace(0, np.nan, inplace=True)
knn_imp = KNNImputer(n_neighbors=3)#looks at the nearest 3 values
df_imp = pd.DataFrame(knn_imp.fit_transform(data), columns=data.columns)

#Display imputed values
print("Imputed Dataset\n")
print(df_imp.head(25))

#Example dataframe.iloc[19, 3] was 0 now imputed to -0.20.
print("Imputed Value: ", df_imp.iloc[20, 3])

#3) FEATURE ENGINEERING
print("\n")

#Rename Variable Columns to Season names
df_imp = df_imp.rename(columns={'J-D': 'Annual_Average','DJF': 'SUMMER', 'MAM': 'AUTUMN', 'JJA' : 'WINTER', 'SON' : 'SPRING'})

#Create new column - 5-Year-Moving-Avg: Capture long-term trends.
#Calculate 5 year Average of the yearly average temperature

df_imp['5Year_Avg'] = df_imp['Annual_Average'].rolling(window=5, min_periods=1).mean()

print(df_imp.head(5))
print(df_imp.columns)

# Write the DataFrame to a CSV file
df_imp.to_csv('Cleaned_Data.csv', index=False)