# Machine Learning Algorithms Implementation

A comprehensive collection of fundamental Machine Learning algorithms implemented from scratch and using Scikit-Learn. 
This repository serves as a practical guide for understanding the mathematical foundations and implementation details of various supervised learning models.

---

## Algorithms Included

###  Regression Models
* **Linear Regression:** Implementation of Simple Linear Regression to predict a continuous target variable based on a single feature.
* **Multiple Linear Regression:** Predicting outcomes using multiple independent variables, accounting for multicollinearity and feature scaling.
* **Polynomial Linear Regression:** Modeling non-linear relationships by transforming features into polynomial terms.
* **Support Vector Regression (SVR):** Using Support Vector Machines to predict continuous values by finding a hyperplane that fits the data within a specific threshold.
* **K-Nearest Neighbors (KNN) Regression:** Predicting values based on the average (or weighted average) of the $k$ most similar neighboring data points.

### Classification Models
* **Logistic Regression:** A foundational binary classification algorithm used to estimate the probability of a class membership.
* **Support Vector Machine (SVM):** Finding the optimal hyperplane that maximizes the margin between different classes. Supports linear and non-linear (kernel-based) classification.

* **Naive Bayes:** A probabilistic classifier based on Bayes' Theorem with the "naive" assumption of conditional independence between every pair of features.
* **K-Nearest Neighbors (KNN) Classification:** Classifying data points based on the majority vote of their nearest neighbors in the feature space.


---

## Requirements
* Python 3.x
* NumPy
* Pandas
* Matplotlib / Seaborn
* Scikit-Learn

---

## Usage
Each directory contains a Jupyter Notebook or Python script demonstrating:
1. **Data Preprocessing:** Handling missing values, Encoding, and Feature Scaling.
2. **Model Training:** Fitting models to training data.
3. **Hyperparameter Tuning:** Logic to iterate through various $k$ values ($1$ to $20$) to visualize and select the highest accuracy score.
4. **Performance Evaluation:** Utilizing Accuracy Score, R-squared ($R^2$), and Confusion Matrices.

---

## Repository Structure
* `/Linear_Regression` - Simple, Multiple, and Polynomial implementations.
* `/Logistic_Regression` - Classification tasks.
* `/SVM` - Scripts for both SVR and SVC.
* `/Naive_Bayes` - Categorical and Gaussian implementations.
* `/KNN` - Classification and Regression examples.

---
