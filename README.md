# Auto Insurance Claim Fraud Detection

Detecting fraudulent insurance claims using machine learning — achieving 93.3% accuracy and 0.84 ROC-AUC with XGBoost on a highly imbalanced dataset.


# Problem Statement
Insurance fraud costs the industry billions annually. Manual review of every claim is slow and error-prone. This project builds an automated fraud detection system that scores each claim and flags high-probability fraud cases — enabling investigators to focus where it matters most.
Business Question: Can we predict whether an insurance claim is fraudulent before it is paid out?

# Dataset
PropertyDetailSourceAuto Insurance Claims DatasetRecords15,420 claimsFeatures33 (vehicle info, driver history, claim details)Class Distribution923 fraud (6%) vs 14,497 legitimate (94%)

# Approach
Raw Data → EDA → Feature Engineering → Class Imbalance Handling (SMOTE) → Model Training → Evaluation
Key Steps

Exploratory Data Analysis — identified VehicleAge and PastNumberOfClaims as the strongest fraud predictors
Class Imbalance — resolved severe imbalance (6% fraud) using SMOTE oversampling → balanced training set of ~23,000 samples
Model Comparison — trained and evaluated Logistic Regression, Random Forest, and XGBoost
Hyperparameter tuning — optimized XGBoost for precision-recall trade-off


# Results
ModelTest AccuracyROC-AUCNotesLogistic Regression78.2%0.71BaselineRandom Forest91.1%0.79Overfit on training dataXGBoost 93.3%0.84Best overall performance
Top Fraud Predictors: Vehicle Age · Past Number of Claims · Policy Deductible · Incident Severity

# Tech Stack
Python Pandas NumPy Scikit-learn XGBoost imbalanced-learn (SMOTE) Matplotlib Seaborn Jupyter Notebook

# Business Impact

- Automates risk scoring across thousands of claims simultaneously
- Prioritizes investigator workload toward highest-probability fraud
- Reduces false negatives that allow fraudulent payouts to slip through
