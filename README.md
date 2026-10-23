# Stroke Prediction Project

A machine learning project that predicts the likelihood of a patient having a stroke from demographic and health data.

## Overview

Stroke is one of the leading causes of death and disability worldwide. This project explores patient data, builds classification models, and evaluates how well they identify patients at risk. It is for educational purposes only and is **not a medical diagnostic tool**.

## Dataset

- **Source:** [e.g., Kaggle "Stroke Prediction Dataset"]
- **Size:** [number of rows] patients, [number of columns] features
- **Features:** gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, smoking status
- **Target:** `stroke` (1 = stroke, 0 = no stroke)

## Project Structure

```
Stroke-Project/
├── data/            # Raw and processed datasets
├── notebooks/       # Exploratory analysis and modeling notebooks
├── src/             # Scripts for preprocessing, training, evaluation
├── models/          # Saved trained models
├── requirements.txt # Dependencies
└── README.md
```

## Methodology

1. **Data cleaning:** handling missing values (e.g., BMI) and encoding categorical variables
2. **Exploratory data analysis:** distributions, correlations, and risk-factor trends
3. **Class imbalance handling:** [e.g., SMOTE / class weights / undersampling], since strokes are rare in the data
4. **Modeling:** [e.g., Logistic Regression, Random Forest, XGBoost]
5. **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC (recall matters most here, since missing a true stroke case is costly)

## Results

| Model | Accuracy | Recall | F1 | ROC-AUC |
|-------|----------|--------|-----|---------|
| [Model 1] | [ ] | [ ] | [ ] | [ ] |
| [Model 2] | [ ] | [ ] | [ ] | [ ] |

**Key findings:** [e.g., age, glucose level, and hypertension were the strongest predictors.]

## Getting Started

```bash
git clone https://github.com/mayss5a/Stroke-Project.git
cd Stroke-Project
pip install -r requirements.txt
jupyter notebook
```

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## Limitations

- The dataset is small and highly imbalanced.
- Results may not generalize to real clinical populations.
- Not intended for medical use.

## Author

**mayss5a** — [GitHub](https://github.com/mayss5a)

## License

[e.g., MIT]
