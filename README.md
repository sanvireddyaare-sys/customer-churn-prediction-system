# Customer Churn Prediction System

An end-to-end Machine Learning project that predicts customer churn using a tuned XGBoost classifier. The system estimates the probability of a customer leaving a telecom service and provides insights to support proactive customer retention.

## Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave helps companies take preventive action and improve customer retention.

This project uses historical telecom customer data to train and evaluate machine learning models and provides an interactive web application for predicting churn risk.

## Key Features

- **Data Analysis:** Explore customer demographics, service usage, contracts, and churn patterns.
- **Data Preprocessing:** Handle missing values, scale numerical features, and encode categorical variables.
- **Model Comparison:** Evaluate Logistic Regression, Random Forest, and tuned XGBoost.
- **Hyperparameter Tuning:** Use cross-validation to identify suitable XGBoost parameters.
- **Threshold Optimization:** Adjust the classification threshold to balance churn precision and recall.
- **Model Explainability:** Use SHAP to understand feature importance and how customer attributes influence predictions.
- **Interactive Web App:** Enter customer details and obtain churn and retention probability estimates.
- **Model Persistence:** Save the trained model, preprocessing pipeline, and decision threshold for reuse.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

## Machine Learning Models

Three classification models are evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. Tuned XGBoost Classifier

The models are compared using accuracy, precision, recall, F1-score, and ROC-AUC. The tuned XGBoost model is selected for the application, with a decision threshold of 0.35 used for churn classification.

### Model Evaluation

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 73.81% | 0.8413 |
| Random Forest | 78.28% | 0.8220 |
| Tuned XGBoost | 80.55% | 0.8468 |

*These are the reported initial evaluation results. Evaluate the final saved model on a held-out test set before presenting the results as final.*

## Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to investigate which features influence churn predictions.

The analysis highlights factors such as:

- Contract type
- Customer tenure
- Monthly charges
- Online security
- Internet service type

These insights can help businesses understand churn patterns and develop targeted customer retention strategies.

## Project Structure

```text
Customer_churn_prediction_System/
│
├── 01_Dataset_Exploration.ipynb
├── Untitled.ipynb
├── app.py
├── Telco-Customer-Churn.csv
├── artifacts/
│   ├── churn_model.pkl
│   ├── preprocessor.pkl
│   └── churn_threshold.pkl
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sanvireddyaare-sys/customer-churn-prediction-system.git
cd customer-churn-prediction-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib joblib streamlit
```

### 4. Run the application

Ensure `app.py` and the required model artifacts are present in the project directory.

```bash
python -m streamlit run app.py
```

The application should open in your browser. If it does not, open the local URL displayed in the terminal.

## How to Use

1. Open the Streamlit application.
2. Enter the customer's demographic, contract, and service details.
3. Click **Predict Churn**.
4. Review the estimated churn probability and retention probability.
5. Use the prediction as an indicator for further customer retention analysis.

## Future Enhancements

- Add customer-level explanations directly to the web application.
- Introduce risk categories such as Low, Medium, and High.
- Recommend personalized retention strategies based on the strongest churn drivers.
- Add a business dashboard showing churn trends and customer segments.
- Deploy the application publicly using Streamlit Community Cloud.
- Monitor model performance and data drift as new customer data becomes available.

## Disclaimer

Predictions are estimates based on historical data and model assumptions. They are not guarantees of future customer behavior. The selected classification threshold should be validated against business objectives.

## Author

**Sanvi Reddy**  
B.Tech — Computer Science and Engineering

GitHub: [sanvireddyaare-sys](https://github.com/sanvireddyaare-sys)
