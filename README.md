
# 🏦 Bank Marketing Prediction using Machine Learning

## 1. Project Overview

This project develops a machine learning system to predict whether a bank customer will subscribe to a term deposit.

The project uses customer demographic, financial, campaign, and contact-related information to build classification models.

The project also provides an interactive Streamlit web application where users can upload a CSV file, select a machine learning model, generate predictions, and evaluate model performance.

---

## 2. Problem Statement

The objective is to predict the target variable `y`, which indicates whether a customer subscribed to a bank term deposit.

- `yes` → Customer subscribed
- `no` → Customer did not subscribe

This is a binary classification problem.

---

## 3. Dataset Features

The following features are used:

- age
- job
- marital
- education
- default
- balance
- housing
- loan
- contact
- day
- month
- duration
- campaign
- pdays
- previous
- poutcome

Target variable:

- `y`

---

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset.
2. Separated input features and target variable.
3. Identified numerical and categorical features.
4. Applied appropriate preprocessing to categorical features.
5. Transformed the training and testing data using the preprocessing pipeline.
6. Converted the target labels into a consistent binary representation.
7. Used the same saved preprocessing pipeline in the Streamlit application.

The processed training dataset contains:

**36,168 samples and 51 processed features.**

---

## 5. Machine Learning Models

Five classification algorithms were implemented:

### 5.1 Logistic Regression

A linear classification algorithm used as a baseline model.

### 5.2 Decision Tree

A tree-based model that makes decisions using feature-based splitting.

### 5.3 K-Nearest Neighbors (KNN)

A distance-based classification algorithm that predicts a class based on nearby training observations.

### 5.4 Gaussian Naive Bayes

A probabilistic classification algorithm based on Bayes' theorem and the assumption of conditional independence between features.

### 5.5 Random Forest

An ensemble learning method that combines multiple decision trees to improve prediction performance.

---

## 6. Model Evaluation

The models are evaluated using the following metrics:

### Accuracy

Measures the overall proportion of correctly classified observations.

### AUC

Measures the model's ability to distinguish between the two classes.

### Precision

Measures the proportion of predicted positive cases that are actually positive.

### Recall

Measures the proportion of actual positive cases correctly identified by the model.

### F1 Score

Provides a balance between precision and recall.

### Matthews Correlation Coefficient (MCC)

Measures the quality of binary classifications using all four confusion-matrix categories.

Additional evaluation outputs include:

- Classification Report
- Confusion Matrix

---

## 7. Streamlit Application

The project includes an interactive Streamlit application.

The application provides:

- CSV file upload
- Machine learning model selection
- Customer predictions
- Prediction labels
- Prediction CSV download
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Classification Report
- Confusion Matrix

---

## 8. Project Structure

```text
Bank-Marketing-ML/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── preprocessor.pkl
