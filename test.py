import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['stroke'].value_counts())
sns.countplot(x='stroke', data=df)
plt.title("Stroke Distribution")
plt.show()
print(df['gender'].value_counts())
print(df['smoking_status'].value_counts())
print(df['work_type'].value_counts())
print(df['age'].hist())
plt.title("Age Distribution")
plt.show()
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()
plt.title("Correlation Matrix")
plt.show()
print(numeric_df.head())
print(numeric_df.columns)
print(numeric_df.shape)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(df.dtypes)
numeric_df = df.select_dtypes(include=['int64','float64'])
print(numeric_df.shape)
numeric_df = numeric_df.fillna(numeric_df.mean())
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()