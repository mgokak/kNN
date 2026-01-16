# k-Nearest Neighbors (kNN) – Classification & Regression

## Overview

This repository contains Jupyter Notebooks demonstrating the **k-Nearest Neighbors (kNN)** algorithm for both **classification** and **regression** tasks. The notebooks focus on understanding how kNN works, how distance-based learning is performed, and how the choice of *k* impacts model performance.

---

## Table of Contents

1. Installation  
2. Project Structure  
3. kNN for Classification  
4. kNN for Regression  
5. Choosing the Value of k  
6. Model Evaluation  

---

## Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

- `kNN_Classification.ipynb` – kNN applied to classification problems  
- `KNN_Regressor.ipynb` – kNN applied to regression problems  

---

## kNN for Classification

### `kNN_Classification.ipynb`

This notebook applies kNN to **classification problems**, where predictions are made based on the most common class among the nearest neighbors.

Key points:
- Distance-based, non-parametric algorithm
- No explicit training phase
- Sensitive to feature scaling

Basic commands used:
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

---

## kNN for Regression

### `KNN_Regressor.ipynb`

This notebook demonstrates **kNN regression**, where predictions are made by averaging the values of nearest neighbors.

Key points:
- Used for continuous target prediction
- Predictions depend heavily on local neighborhoods
- Performance affected by noise and outliers

Basic commands used:
```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)
```

---

## Choosing the Value of k

Key points:
- Small *k* → low bias, high variance
- Large *k* → high bias, low variance
- Optimal *k* is usually found through experimentation

Common approach:
```python
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
```

---

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, r2_score
```

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  
DePaul University
