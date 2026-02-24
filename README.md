<div align="center">

# Machine Learning Algorithms Implementation

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)](https://numpy.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)](#-contributing)

**A comprehensive collection of fundamental Machine Learning algorithms implemented from scratch and using Scikit-Learn.**
This repository serves as a practical guide for understanding the mathematical foundations, intuition, and implementation details of various supervised learning models.

---

</div>

## Algorithms at a Glance

| Regression ( Predict a Number )   | Classification ( Predict a Category ) |
| --------------------------------- | ------------------------------------- |
| • Linear Regression               | • Logistic Regression                 |
| • Multiple Linear Regression      | • Support Vector Machine (SVM)        |
| • Polynomial Regression           | • Naive Bayes                         |
| • Support Vector Regression (SVR) | • K-Nearest Neighbors (KNN)           |
| • KNN Regression                  |                                       |

| Type           | Algorithm                  | Key Idea                        |
| :------------- | :------------------------- | :------------------------------ |
| Regression     | Linear Regression          | Best-fit straight line          |
| Regression     | Multiple Linear Regression | Multiple features → one target  |
| Regression     | Polynomial Regression      | Curved best-fit line            |
| Regression     | SVR                        | Fit within error margin         |
| Regression     | KNN Regression             | Average of nearest neighbors    |
| Classification | Logistic Regression        | Probability of class membership |
| Classification | SVM                        | Maximum margin hyperplane       |
| Classification | Naive Bayes                | Bayes' Theorem + independence   |
| Classification | KNN Classification         | Majority vote of neighbors      |

---

## Regression Models

* **Goal:** Predict a **continuous numerical** value (e.g., price, salary, temperature).

---

### 1️⃣ Linear Regression

* Finds the **best-fit straight line** through the data to predict a continuous target from a single feature.

#### Core Equation

* ŷ = β₀ + β₁x
* Where:

  * ŷ = Predicted value
  * β₀ = Intercept (y-axis crossing)
  * β₁ = Slope (rate of change)
  * x = Input feature

```
y
│
│        •
│      •
│    •
│  •
│ •───────────────        ← Regression line (ŷ = β₀ + β₁x)
│   │
│   │ residual (error)
│   •
│     •
└────────────────────────── x
```

#### ✅ When to Use

* **Linear relationship** between feature and target
* Single input feature
* Quick baseline model

#### ⚠️ Limitations

* Cannot capture curved or complex patterns
* Sensitive to **outliers**
* Assumes **constant variance** of errors (homoscedasticity)

---

### 2️⃣ Multiple Linear Regression

* Extends linear regression to **multiple independent variables**, predicting outcomes based on several features simultaneously.

#### Core Equation

ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + βₙxₙ

##### Example:

* Salary = β₀ + β₁(Experience) + β₂(Education) + β₃(Age)

#### ✅ When to Use

* Multiple features influence the target
* Features are **not highly correlated** with each other

#### ⚠️ Watch Out For

| Issue                   | What It Means                           | Solution                                 |
| :---------------------- | :-------------------------------------- | :--------------------------------------- |
| **Multicollinearity**   | Features are correlated with each other | Remove or combine correlated features    |
| **Irrelevant features** | Noise features hurt performance         | Feature selection (backward elimination) |
| **Different scales**    | Large-valued features dominate          | Apply feature scaling                    |

---

### 3️⃣ Polynomial Linear Regression

* Models **non-linear (curved) relationships** by transforming features into polynomial terms while still using a linear model framework.

#### Core Equation

ŷ = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ

* Degree 1: Straight line ──────
* Degree 2: Parabola ╱╲
* Degree 3: S-curve ╱╲╱

#### ✅ When to Use

* Data shows a **curved / non-linear trend**
* Linear regression gives poor results (low R²)

#### ⚠️ Watch Out For

* **Overfitting** with high-degree polynomials — the model memorizes noise
* Always **validate** with a test set or cross-validation

---

### 4️⃣ Support Vector Regression (SVR)

* Uses Support Vector Machines to predict continuous values by fitting data within an **ε-insensitive tube** (error margin).

#### Core Concept

* Points OUTSIDE the tube = Support Vectors (they define the model)
* Points INSIDE the tube = No penalty

#### ✅ When to Use

* Data has **outliers** (SVR is robust to them)
* Non-linear relationships (with **RBF / polynomial kernels**)
* Medium-sized datasets

#### Key Hyperparameters

| Parameter     | Role                                                      |
| :------------ | :-------------------------------------------------------- |
| `C`           | Regularization — trade-off between error and margin width |
| `ε` (epsilon) | Width of the insensitive tube                             |
| `kernel`      | Transformation function (`linear`, `rbf`, `poly`)         |

#### ⚠️ Important

* SVR **requires feature scaling** — always standardize your features before training.

---

### 5️⃣ K-Nearest Neighbors (KNN) Regression

> Predicts a value based on the **average (or weighted average)** of the `k` most similar neighboring data points.

#### Core Concept

* Prediction = Average of neighbor values
  = (y₁ + y₂ + y₃) / 3

#### ✅ When to Use

* **Non-linear** relationships
* Small to medium datasets
* When you need a simple, intuitive model

#### Choosing the Right `k`

* **Tip:** Iterate through `k = 1 to 20`, plot accuracy, and select the optimal value.

#### ⚠️ Watch Out For

* **Requires feature scaling** — distance-based algorithm
* Slow on **large datasets** (computes distances to every point)
* Sensitive to **irrelevant features**

---

## Classification Models

- **Goal:** Predict a **discrete category/class** (e.g., spam/not spam, disease/healthy, yes/no).

---

### 1️⃣ Logistic Regression

- A foundational **binary classification** algorithm that estimates the **probability** of class membership using the sigmoid function.

####  Core Equation

```
             1
P(y = 1) = ─────────────
           1 + e^(-(β₀ + β₁x))
```

**Output:** Probability between 0 and 1
**Decision Rule:** If `P ≥ 0.5` → Class 1, else → Class 0

---

#### Sigmoid Function

```
P(y=1)
1.0 │        ─────────
    │      /
0.5 │ · · · · · · / · · · · · ·   ← Decision boundary
    │    /
0.0 │──────────────
    └──────────────────────── x
```

---

#### ✅ When to Use

* Binary classification (two classes)
* Linearly separable data
* When you need **probability estimates**
* Fast, interpretable baseline model

#### ⚠️ Limitations

* Assumes a **linear decision boundary**
* Struggles with complex, non-linear patterns

---

### 2️⃣ Support Vector Machine (SVM)

- Finds the **optimal hyperplane** that maximizes the **margin** between different classes. Supports linear and non-linear (kernel-based) classification.

#### Core Concept

```
Class A: ○      Class B: ●

○  ○              ●  ●
  ○  ○    ┃     ●  ●
    ○  ○  ┃   ●  ●        ← Maximum margin hyperplane
      ○  ◁┃▷ ●
    ○  ○  ┃   ●  ●
  ○  ○    ┃     ●  ●
          ┃
     ◁────┃────▷
        MARGIN (maximized)

◁▷ = Support Vectors
```

---

#### Kernel Trick

- For **non-linearly separable** data, kernels transform features into a higher-dimensional space where a linear boundary can be found.

| Kernel           | Use Case                 | Boundary                   |
| ---------------- | ------------------------ | -------------------------- |
| `linear`         | Linearly separable data  | Straight line / plane      |
| `rbf` (Gaussian) | Most non-linear problems | Flexible curved boundary   |
| `poly`           | Polynomial boundaries    | Curved with degree control |

---

#### ✅ When to Use

* High-dimensional data
* Clear margin of separation
* Binary or multi-class classification

#### ⚠️ Important

- SVM **requires feature scaling** — always standardize before training.

---

### 3️⃣ Naive Bayes

- A **probabilistic classifier** based on **Bayes’ Theorem** with the assumption that features are **conditionally independent** given the class.

#### Bayes' Theorem

```
              P(X | C) · P(C)
P(C | X) = ─────────────────────
                P(X)
```

Where:

* **P(C | X)** → Posterior probability
* **P(X | C)** → Likelihood
* **P(C)** → Prior probability
* **P(X)** → Evidence

---

#### Intuition Example

**Is this email SPAM?**

Features:

* Contains "free"
* Contains "winner"
* Length > 100

We compute:

```
P(Spam | features) vs P(Not Spam | features)
```

Choose the class with **higher probability** ✅

---

#### Variants

| Variant        | Feature Type | Example        |
| -------------- | ------------ | -------------- |
| Gaussian NB    | Continuous   | Age, salary    |
| Multinomial NB | Count data   | Word frequency |
| Bernoulli NB   | Binary (0/1) | Word presence  |

---

#### ✅ When to Use

* Text classification (spam, sentiment)
* Small datasets
* Real-time prediction (very fast)

#### ⚠️ Limitations

* Independence assumption rarely true
* Complex models may outperform on large datasets

---

### 4️⃣ K-Nearest Neighbors (KNN)

- Classifies data points based on the **majority vote** of their `k` nearest neighbors.

####  Core Concept

```
○  ○        ●
  ○  ◎        ●  ●

    ◎  ★  ◎        ○ = Class A
  ○       ●  ●     ● = Class B
○           ●      ◎ = Neighbors

k = 3 → 2 Class A, 1 Class B
Prediction = Class A ✅
```

---

#### Choosing the Right k

| k Value          | Effect                   |
| ---------------- | ------------------------ |
| Too small (k=1)  | Overfitting              |
| Too large (k=50) | Underfitting             |
| Optimal          | Best validation accuracy |

---

#### Hyperparameter Tuning Example

```python
from sklearn.neighbors import KNeighborsClassifier

for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"k={k}: Accuracy = {accuracy:.4f}")
```

---

#### ✅ When to Use

* Non-linear decision boundaries
* Small to medium datasets
* Multi-class classification

#### ⚠️ Watch Out

* Requires feature scaling
* Slow on large datasets
* Curse of dimensionality

---

#  Standard Workflow Used in All Algorithms

```
┌─────────────────────────────────────────────────────────────┐
│                    STANDARD WORKFLOW                         │
├──────────┬──────────────────────────────────────────────────┤
│ Step 1   │  Data Preprocessing                            │
│          │ • Handling missing values                        │
│          │ • Encoding categorical variables                 │
│          │ • Feature scaling                                │
├──────────┼──────────────────────────────────────────────────┤
│ Step 2   │ Train-Test Split                               │
│          │ • 80/20 split                                    │
├──────────┼──────────────────────────────────────────────────┤
│ Step 3   │  Model Training                                │
│          │ • Fit model on training data                     │
├──────────┼──────────────────────────────────────────────────┤
│ Step 4   │  Hyperparameter Tuning                         │
│          │ • K selection (KNN)                              │
│          │ • Kernel selection (SVM)                         │
│          │ • Degree selection                               │
├──────────┼──────────────────────────────────────────────────┤
│ Step 5   │  Performance Evaluation                         │
│          │ • Regression → R², MSE, MAE                      │
│          │ • Classification → Accuracy, Confusion Matrix    │
│          │ • Visualization                                  │
└──────────┴──────────────────────────────────────────────────┘
```

---

 
