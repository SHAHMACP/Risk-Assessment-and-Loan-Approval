# 🏦 Loan Approval Prediction Project
Synthetic Dataset for Risk Assessment and Loan Approval Modeling(Classification and Regression)
This synthetic dataset comprises 20,000 records of personal and financial data, designed to facilitate the development of predictive models for risk assessment. It serves two primary purposes: The dataset includes diverse features such as demographic information, credit history, employment status, income levels, existing debt, and other relevant financial metrics, providing a comprehensive foundation for sophisticated data-driven analysis and decision-making.
![image](https://github.com/user-attachments/assets/024780ba-6e2f-421c-9795-52907109a77c)

> Risk Score Regression: To predict a continuous risk score associated with each individual's likelihood of loan default or financial instability.

> Binary Classification: To determine the binary outcome of loan approval, indicating whether an applicant is likely to be approved or denied for a loan.



---

## 📁 Dataset
- File: `Loan.csv`:  https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval/data.
- Contains applicant's financial, employment, and loan-related data.

---

## ✅ Objective
- Predict `RiskScore` using LGBMRegressor
- Predict `LoanApproved` using RandomForestClassifier

---

## 🔧 Data Preprocessing & Feature Engineering

### 1. Dropped Unnecessary Columns
- `AnnualIncome`, `TotalAssets`, `TotalLiabilities`, `DebtToIncomeRatio`, `TotalDebtToIncomeRatio`, etc.

### 2. Created Features
- `BankBalance = SavingsAccountBalance + CheckingAccountBalance`
- Extracted `year`, `month`, `day` from `ApplicationDate` and dropped original

### 3. Encoding Categorical Features
- **Ordinal Encoding**: `EmploymentStatus`, `EducationLevel`, `HomeOwnershipStatus`
- **Target Encoding**: `MaritalStatus`
- **Frequency Encoding**: `LoanPurpose`

### 4. Skewness Handling
- Applied log transformation to highly skewed numerical features

### 5. Outlier Removal
- Used IQR method to remove outliers from numeric columns

### 6. Scaling
- Standardized all numeric features using `StandardScaler`

---

## 📈 Model 1: LGBM Regressor (Predicting Risk Score)

### 🔹 Features: 
- All numeric columns (excluding `LoanApproved` and `RiskScore`)

### 🔹 Target:
- `RiskScore`

### 🔹 Metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

### 🔹 Plots:
- Actual vs Predicted Scatter Plot
- Residuals Distribution

---

## 📊 Model 2: Random Forest Classifier (Predicting Loan Approval)

### 🔹 Feature:
- `RiskScore`

### 🔹 Target:
- `LoanApproved` (converted to int)

### 🔹 Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

### 🔹 Plots:
- Confusion Matrix
- Predicted Loan Approval Probability vs. Risk Score

---
## 🧠 Results
> Based on the analysis of the loan approval dataset for predicting the RiskScore, the **LGBMRegressor** (on Scaled data)	 model demonstrated the best performance among the tested models.

> Based on the evaluation of the classification models for predicting loan approval, the **RandomForestClassifier()** emerged as the best-performing model. It achieved the highest accuracy and consistently strong results across precision, recall, and F1-score, making it the most reliable model for classifying whether a loan is approved or not in the loan dataset.
![image](https://github.com/user-attachments/assets/d77243aa-78a1-40a2-936c-0e830973fd09)

---
## 🧠 Future Improvements
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Use more features to predict `LoanApproved`
- Try XGBoost, CatBoost, or Logistic Regression

---

## 👩‍💻 Author
This project was developed as part of a data science learning module focused on regression and classification modeling.

---


