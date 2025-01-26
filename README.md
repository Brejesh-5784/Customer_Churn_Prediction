### **Customer Churn Prediction - Machine Learning Workflow Documentation**

#### **Overview:**
This document outlines the workflow for predicting customer churn using machine learning techniques. The process involves loading the dataset, performing exploratory data analysis (EDA), preprocessing the data, training two different machine learning models (Random Forest and XGBoost), evaluating their performance, and saving the trained models for future use.

---

### **Libraries Used:**

- **pandas**: Used for data manipulation and analysis.
- **numpy**: Provides support for numerical computing.
- **matplotlib.pyplot**: Used for creating static, animated, and interactive visualizations.
- **seaborn**: A library for statistical data visualization built on top of matplotlib.
- **sklearn**: A machine learning library that includes models, metrics, and utilities for classification.
- **xgboost**: An implementation of gradient boosted decision trees designed for speed and performance.
- **joblib**: Used for saving and loading trained machine learning models.

---

### **Steps and Code Walkthrough:**

#### **Step 1: Load Train and Test Datasets**
- The datasets for training and testing the model are loaded from CSV files.
- Basic data inspection is performed to understand the structure and check for missing values.

#### **Step 2: Exploratory Data Analysis (EDA)**
- **Churn Distribution**: The distribution of the target variable, `Churn`, is visualized to assess the class balance.
- **Churn vs Tenure**: A boxplot is used to examine the relationship between `Churn` and `Tenure` to see how customer tenure correlates with churn.
- **Correlation Heatmap**: A heatmap is used to display correlations between numerical features to identify strong relationships.

#### **Step 3: Data Preprocessing**
- **Handling Missing Values**: Missing values in the `Total Spend` column are handled by converting the column to numeric and filling missing values with the median.
- **Dropping Irrelevant Columns**: The `CustomerID` column is dropped since it doesn’t contribute to the prediction.
- **Encoding Categorical Variables**: Categorical features are converted into dummy variables to make them suitable for model training.
- **Aligning Train and Test Data**: The target variable `Churn` is separated from the features for both training and testing datasets.

#### **Step 4: Train the Random Forest Model**
- A Random Forest model is trained on the preprocessed training data.
- Model predictions are made on the test dataset, and performance is evaluated using metrics such as precision, recall, F1-score, and support for each class.

#### **Step 5: Train the XGBoost Model**
- Similarly, an XGBoost model is trained on the preprocessed data.
- The performance of the XGBoost model is evaluated using the same classification metrics.

#### **Step 6: Confusion Matrix and F1-Score**
- A confusion matrix is displayed for both the Random Forest and XGBoost models to visualize the true positives, false positives, true negatives, and false negatives.
- The F1-score, which balances precision and recall, is calculated for both models to assess their overall performance.

#### **Step 7: Feature Importance (Random Forest)**
- The Random Forest model's feature importance is evaluated, and the most important features are visualized to understand which factors influence churn predictions the most.

#### **Step 8: Save the Models**
- Both trained models (Random Forest and XGBoost) are saved using `joblib`, allowing for future use without retraining.

---

### **Model Evaluation:**
- **Classification Report**: A report summarizing key metrics (precision, recall, F1-score, support) for both models is generated.
- **Confusion Matrix**: A matrix for each model is displayed to show the performance of the models in terms of true and false positives/negatives.
- **F1-Score**: The F1-score is calculated for each model to measure the balance between precision and recall.

---

### **Important Notes:**
- **Preprocessing**: The dataset undergoes preprocessing steps to handle missing values and encode categorical variables, ensuring the data is ready for training.
- **Model Training**: Two machine learning models (Random Forest and XGBoost) are trained on the dataset to predict customer churn.
- **Feature Importance**: The importance of features in making churn predictions is evaluated using the Random Forest model, helping to identify which features are most influential.
- **Model Persistence**: Both trained models are saved to disk using `joblib` to enable easy reloading and inference in the future.

---

This workflow provides a structured approach to building a customer churn prediction system, covering all steps from data preprocessing and exploratory analysis to model training, evaluation, and saving the models for deployment.
