import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA

df = pd.read_csv("healthcare-dataset-stroke-data.csv")


# BASIC EXPLORATION

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

#DATA PREPROCESSING
df = df.drop('id', axis=1, errors='ignore')
print(df.columns)

df = pd.get_dummies(df, drop_first=True)

X = df.drop('stroke', axis=1)
y = df['stroke']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# step 5 building models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("LOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

print("\nKNN")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

from sklearn.metrics import roc_auc_score

print("ROC-AUC (RF):", roc_auc_score(y_test, y_pred_rf))

#STEP 6 — FEATURE IMPORTANCE & INTERPRETATION
import pandas as pd

feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print(feature_importance)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
feature_importance.plot(kind='bar')
plt.title("Feature Importance (Random Forest)")
plt.show()

feature_importance.head(3)
feature_importance.head(3).plot(kind='bar')
plt.title("Top 3 Important Features")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Probabilities
y_prob_log = log_model.predict_proba(X_test)[:,1]

# ROC values
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)

# Plot
plt.figure()
plt.plot(fpr_log, tpr_log, label="Logistic Regression")
plt.plot([0,1], [0,1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()

# AUC
print("AUC Logistic Regression:", roc_auc_score(y_test, y_prob_log))

# Probabilities
y_prob_knn = knn_model.predict_proba(X_test)[:,1]

# ROC values
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)

# Plot
plt.figure()
plt.plot(fpr_knn, tpr_knn, label="KNN")
plt.plot([0,1], [0,1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN")
plt.legend()
plt.show()

# AUC
print("AUC KNN:", roc_auc_score(y_test, y_prob_knn))

#STEP 7 — MODEL COMPARISON
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'KNN'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_knn)
    ],
    'Recall': [
        recall_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_knn)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_knn)
    ]
})

print(results)

#fixing the imbalance
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_train, y_train = sm.fit_resample(X_train, y_train)

print(y_train.value_counts())

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)

y_prob_rf = rf_model.predict_proba(X_test)[:,1]

# Change threshold from 0.5 → 0.3
y_pred_rf = (y_prob_rf > 0.3).astype(int)

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

from sklearn.metrics import accuracy_score, recall_score, f1_score

print("Recall RF:", recall_score(y_test, y_pred_rf))

#Compare models after resolving the imbalance
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'KNN'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_knn)
    ],
    'Recall': [
        recall_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_knn)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_knn)
    ]
})

print(results)

#STEP 8 — CONFUSION MATRIX VISUALIZATION
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#STEP 9 — ROC CURVE & AUC
from sklearn.metrics import roc_curve, roc_auc_score

y_prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0,1], [0,1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("AUC:", roc_auc_score(y_test, y_prob))

import joblib

# Save model
joblib.dump(rf_model, "rf_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

joblib.dump(X.columns, "features.pkl")