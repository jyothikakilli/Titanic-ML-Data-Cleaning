# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('mldata.csv')

# Quick look at the data
print(data.head())
print(data.info())
print(data.describe())

# Handling missing values
# Option 1: Drop missing values
data = data.dropna()

# Option 2: Fill missing values with mean/median/mode
data['column_name'] = data['column_name'].fillna(data['column_name'].mean())

# Removing duplicates
data = data.drop_duplicates()

# Encoding categorical variables
data = pd.get_dummies(data, drop_first=True)

# Detecting and removing outliers using the IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Data visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.title('Boxplot for Outlier Detection')
plt.show()

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Save the cleaned data
data.to_csv('cleaned_dataset.csv', index=False)

print("Data cleaning and preprocessing completed!")