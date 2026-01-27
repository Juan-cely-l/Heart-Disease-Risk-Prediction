# Heart Disease Risk Prediction

## Repo Overview
**Exercise Summary:** Implements logistic regression for heart disease prediction, including:
- Exploratory Data Analysis (EDA)
- Model training and visualization
- Regularization
- Deployment on AWS SageMaker

---

## About Dataset
**Heart Disease Prediction Dataset**

This dataset contains real-world clinical attributes used to analyze and predict the presence or absence of heart disease.  
Each row represents one patient, and each column represents a medical measurement or diagnostic indicator.

### Suitable for:
- Exploratory Data Analysis (EDA)
- Machine Learning / AI models
- Binary classification
- Feature importance analysis
- Medical data science practice

---

## Column Descriptions (Data Dictionary)

| Column Name              | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Age                      | Age of the patient (in years)                                               |
| Sex                      | Gender of the patient (1 = Male, 0 = Female)                                |
| Chest pain type          | Type of chest pain: 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic |
| BP                       | Resting blood pressure (mm Hg)                                              |
| Cholesterol              | Serum cholesterol level (mg/dL)                                             |
| FBS over 120             | Fasting blood sugar > 120 mg/dL (1 = True, 0 = False)                       |
| EKG results              | Resting electrocardiogram results: 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy |
| Max HR                   | Maximum heart rate achieved                                                 |
| Exercise angina          | Exercise-induced angina (1 = Yes, 0 = No)                                   |
| ST depression            | ST depression induced by exercise relative to rest                          |
| Slope of ST              | Slope of the peak exercise ST segment                                       |
| Number of vessels fluro  | Number of major vessels (0–3) colored by fluoroscopy                        |
| Thallium                 | Thallium stress test result (categorical medical indicator)                 |
| Heart Disease            | Target variable: Presence = Heart disease detected, Absence = No heart disease |

---

## Encoding Notes
- Categorical variables are numerically encoded for ML compatibility.
- Target column uses text labels (Presence / Absence) for better interpretability.
- Dataset is ready for Logistic Regression, Tree-based models, and Ensembles.

---

## Source & Context
This dataset follows standard clinical encodings commonly used in heart disease research, similar to datasets used in:
- Medical machine learning studies
- Academic projects
- Kaggle notebooks & benchmarks

---

## Disclaimer
This dataset is intended **only for educational and research purposes**.  
It must **not** be used for real-world medical diagnosis or treatment decisions without professional clinical validation.

---

## File Information
**Heart_Disease_Prediction.csv** (11.93 kB)  
Contains clinical and demographic features used to predict whether a patient has heart disease.

- Each row represents one individual.
- Each column describes a medical attribute such as age, blood pressure, cholesterol levels, chest pain type, ECG results, and more.

**Target column:**  
- `HeartDisease` — 1 indicates presence of heart disease, 0 indicates absence.

---



## Tags
- Data Analytics  
- Data Visualization  
- Python  
- Logistic Regression  
- XGBoost  

---
