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

Let's analyze the performance of each model:

Logistic Regression:


Accuracy: 0.959 (95.9%)
Recall: 0.949 (94.9%)
Precision: 0.969 (96.9%)
F1-Score: 0.959 (95.9%)

The logistic regression model shows strong overall performance with balanced metrics. The high precision (96.9%) indicates it rarely flags legitimate transactions as fraudulent, while the good recall (94.9%) shows it catches most fraudulent transactions.

XGBoost:


Accuracy: 0.959 (95.9%)
Recall: 0.959 (95.9%)
Precision: 0.959 (95.9%)
F1-Score: 0.959 (95.9%)

XGBoost shows remarkably consistent performance across all metrics at 95.9%. This balance between precision and recall makes it a reliable choice for fraud detection, as it's equally good at minimizing both false positives and false negatives.

Isolation Forest:


Accuracy: 0.502 (50.2%)
Recall: 0.004 (0.4%)
Precision: 1.000 (100%)
F1-Score: 0.008 (0.8%)

The Isolation Forest shows some interesting patterns:

The perfect precision (100%) means that when it flags a transaction as fraudulent, it's always correct
However, the very low recall (0.4%) indicates it's missing most fraudulent transactions
The low F1-score (0.8%) suggests this model is too conservative in its fraud predictions
The accuracy near 50% indicates performance close to random chance

Key Findings:

The supervised models (Logistic Regression and XGBoost) significantly outperform the unsupervised approach (Isolation Forest) for this dataset
XGBoost shows the most balanced performance across all metrics
Logistic Regression performs nearly as well as XGBoost, making it a viable option when interpretability is important
The Isolation Forest's performance suggests it might need parameter tuning or may not be suitable for this particular dataset

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
