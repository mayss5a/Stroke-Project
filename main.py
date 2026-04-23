import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# =========================
# BASIC EXPLORATION
# =========================
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# =========================
# MISSING VALUES
# =========================
print("\nMissing values:\n", df.isnull().sum())

# =========================
# TARGET DISTRIBUTION
# =========================
print("\nStroke distribution:\n", df['stroke'].value_counts())

sns.countplot(x='stroke', data=df)
plt.title("Stroke Distribution")
plt.show()

# =========================
# CATEGORICAL VARIABLES
# =========================
print("\nGender:\n", df['gender'].value_counts())
print("\nSmoking Status:\n", df['smoking_status'].value_counts())
print("\nWork Type:\n", df['work_type'].value_counts())

# =========================
# NUMERICAL DISTRIBUTION
# =========================
plt.figure()
df['age'].hist()
plt.title("Age Distribution")
plt.show()

# =========================
# SELECT NUMERICAL DATA
# =========================
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Handle missing values
numeric_df = numeric_df.fillna(numeric_df.mean())

# =========================
# CORRELATION MATRIX
# =========================
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#cleaning
print(df.isnull().sum())
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
print(df.isnull().sum())
df.drop('id', axis=1, inplace=True)
print(df.columns)
print("\n nbr of duplicates:\n", df.duplicated().sum())
print(df['gender'].unique())
df = df[df['gender'] != 'Other']
print(df['gender'].unique())
df.reset_index(drop=True, inplace=True)