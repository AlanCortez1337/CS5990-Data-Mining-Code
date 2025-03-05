# -------------------------------------------------------------------------
# AUTHOR: Alan Cortez
# FILENAME: pca.py
# SPECIFICATION: This program takes the dataset provided and removes one feature at a time to preform PCA and see which feature removed gives us a higher PC1 variance
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: About 18-24 hours
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
df = pd.read_csv('./heart_disease_dataset.csv')

#Create a training matrix 
#--> add your Python code here
df_features = df.values

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features
#--> add your Python code here
num_features = len(df_features[0])

pca_data = {}
# Run PCA for 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    target_feature = df.columns[i]
    reduced_data = df.drop(target_feature, axis=1)

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pca_data[df.columns[i]] = pca.explained_variance_ratio_[0]


# Find the maximum PC1 variance
# --> add your Python code here
best_removed_feature = ""   
best_pc1 = -1
for feature, pc1 in pca_data.items():
    if pc1 > best_pc1:
        best_pc1 = pc1
        best_removed_feature = feature

#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance found: {best_pc1} when removing {best_removed_feature}")




