# Insurance Fraud Detection using XGBoost

## Overview
This project focuses on detecting fraudulent insurance claims using **XGBoost**. The dataset contains various policyholder and claim details, and the model is trained to classify claims as fraudulent or non-fraudulent.

## Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numeric features.
- **Exploratory Data Analysis (EDA)**: Identifying patterns, correlations, and anomalies in the data.
- **Feature Engineering**: Creating new features such as claim-to-premium ratio and time gaps between key dates.
- **Model Training**: Using **XGBoost** for fraud detection with hyperparameter tuning.
- **Evaluation Metrics**: Accuracy, precision, recall, and ROC-AUC score.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/sourajyoti593/Insurance_fraud_Analysis.git
   ```
2. Navigate to the project folder:
   ```sh
   cd Insurance_fraud_Analysis
   ```
3. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Place the dataset (`insurance_data.xlsx`) in the project directory.
2. Run the model script:
   ```sh
   python Insurance_Fraud_Analysis.py
   ```
3. View the evaluation metrics in the console output.

## Dataset
- **Source**: https://www.kaggle.com/datasets/mastmustu/insurance-claims-fraud-data
- **Target Variable**: `FRAUD_REPORT_FLAG` (1 = Fraud, 0 = Non-Fraud).

## Model Details
- **Algorithm**: XGBoost Classifier
- **Hyperparameters**: 100 estimators, learning rate 0.1, max depth 5
- **Performance Metrics**: Accuracy, Classification Report, ROC-AUC Score

## Contributing
Feel free to fork this repo, make improvements, and submit a pull request.

## License
This project is licensed under the Apache License 2.0.

