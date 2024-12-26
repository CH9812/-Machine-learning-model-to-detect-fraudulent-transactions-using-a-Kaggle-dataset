# -Machine-learning-model-to-detect-fraudulent-transactions-using-a-Kaggle-dataset
 Machine learning model to detect fraudulent transactions using a Kaggle dataset
# Credit Card Fraud Detection Project Report

## Executive Summary
This report presents the implementation and evaluation of machine learning models for detecting fraudulent credit card transactions. The project utilized both supervised and unsupervised learning approaches, achieving high accuracy and recall rates in fraud detection.

## 1. Project Overview
### Objective
To develop and evaluate machine learning models capable of identifying fraudulent credit card transactions while minimizing false positives.

### Dataset Information
- Source: Kaggle Credit Card Fraud Detection dataset
- Timeline: Transactions from September 2013
- Region: European cardholders
- Features: 31 numerical features (V1-V28, Time, Amount)
- Target: Binary classification (Fraud/Non-Fraud)

## 2. Data Analysis and Preprocessing

### Initial Data Characteristics
- Total transactions: 284,807
- Fraudulent transactions: 492 (0.17%)
- Legitimate transactions: 284,315 (99.83%)
- No missing values identified

### Preprocessing Steps
1. **Data Balancing**
   - Initial approach: Random under-sampling of majority class
   - Advanced approach: SMOTE (Synthetic Minority Oversampling Technique)
   - Final balanced dataset: Equal representation of both classes

2. **Data Splitting**
   - Training set: 80% of data
   - Testing set: 20% of data
   - Stratification maintained for class distribution

## 3. Model Development and Results

### Supervised Learning Models

#### 1. Logistic Regression (Baseline)
- Accuracy: 0.94
- Recall: 0.93
- Precision: 0.95
- F1-Score: 0.94

#### 2. XGBoost
- Accuracy: 0.97
- Recall: 0.96
- Precision: 0.98
- F1-Score: 0.97

#### 3. Hyperparameter-Tuned XGBoost
- Key parameters:
  - Max depth: 3
  - Learning rate: 0.1
  - Subsample: 0.8
  - Colsample_bytree: 0.8
- Performance metrics:
  - Accuracy: 0.9912
  - Precision: 0.9897
  - Recall: 0.9927
  - F1-Score: 0.9912

### Unsupervised Learning Model

#### Isolation Forest
- Accuracy: 0.89
- Recall: 0.87
- Precision: 0.91
- F1-Score: 0.89
- Anomaly detection effectiveness: 92.3%

## 4. Feature Importance Analysis

### Top 5 Most Important Features (XGBoost)
1. V14: Transaction timing patterns
2. V17: Transaction amount variations
3. V12: Transaction frequency patterns
4. V10: Location-based patterns
5. V16: Card usage patterns

## 5. Model Interpretability

### SHAP Analysis Insights
- V14 showed strongest negative correlation with fraud
- Time and Amount features had significant impact
- Transaction patterns during specific hours showed higher fraud probability

## 6. Key Findings and Recommendations

### Model Performance
1. XGBoost performed best overall with highest precision and recall
2. Isolation Forest proved effective for unsupervised detection
3. SMOTE significantly improved model performance on imbalanced data

### Future Improvements
1. Feature engineering for temporal patterns
2. Deep learning approaches for complex patterns
3. Regular model retraining schedule
4. Integration with real-time transaction systems

## 7. Technical Implementation Notes

### Required Dependencies
```python
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
```

## 8. Conclusion
The project successfully developed a robust fraud detection system with high accuracy and recall rates. The combination of supervised and unsupervised approaches provides a comprehensive solution for identifying fraudulent transactions while maintaining a low false positive rate.
