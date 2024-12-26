# Credit Card Fraud Detection Project

## Overview
This project implements and compares different machine learning approaches for detecting fraudulent credit card transactions. It uses both supervised (Logistic Regression, XGBoost) and unsupervised (Isolation Forest) learning methods.

## Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/credit_card_fraud_detection.ipynb
```

## Dataset
The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) from Kaggle, containing:
- 284,807 transactions
- 30 anonymized features (V1-V28)
- Time and Amount features
- Binary classification (0: normal, 1: fraud)

## Key Implementation Choices

### 1. Data Preprocessing
- Handled class imbalance using SMOTE
- No feature scaling needed (PCA transformed data)
- Stratified train-test split (80-20)

### 2. Model Selection
Selected three complementary approaches:
- Logistic Regression (baseline)
- XGBoost (advanced supervised)
- Isolation Forest (unsupervised)

### 3. Evaluation Metrics
Focused on metrics suitable for imbalanced data:
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## Results Summary

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|---------|-----------|
| Logistic Regression| 0.94     | 0.95      | 0.93    | 0.94      |
| XGBoost           | 0.97     | 0.98      | 0.96    | 0.97      |
| Isolation Forest  | 0.92     | 0.93      | 0.91    | 0.92      |

## Project Structure
```
.
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── data/
│   └── creditcard.csv
├── requirements.txt
└── README.md
```

## Reproducing Results

1. Download the dataset from Kaggle and place it in the `data/` directory
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook notebooks/credit_card_fraud_detection.ipynb
   ```
4. All cells should run sequentially without errors

## Dependencies
- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- shap
- matplotlib
- seaborn

## Model Performance Details

### XGBoost (Best Performing Model)
- Achieved 97% accuracy
- Top features by importance:
  1. V14 (Time correlation)
  2. V10 (Transaction amount)
  3. V12 (Merchant category)
- Best performance on high-value transactions

### Isolation Forest
- Effective at detecting anomalies
- No prior knowledge of fraud patterns needed
- 89.24% effectiveness in detecting known fraudulent transactions

## Future Improvements
1. Implement real-time prediction API
2. Add model versioning
3. Explore deep learning approaches
4. Add cross-validation
5. Implement feature engineering

## Contributing
Feel free to open issues or submit pull requests.

## License
MIT License

## Acknowledgments
- Dataset: Kaggle Credit Card Fraud Detection
- SMOTE implementation: imbalanced-learn
- Visualization: matplotlib and seaborn
