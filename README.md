# Predict-credit-card-approvals.ipynb

# Credit Card Approval & Default Prediction Pipeline

An end-to-end Machine Learning pipeline implemented in Python utilizing Logistic Regression to predict the likelihood of credit card borrowers not fully paying back their lines of credit. 

This project explores real-world credit risk data, demonstrates robust data preprocessing techniques, addresses severe class imbalance issues, and utilizes interactive diagnostic plots to expose evaluation pitfalls like the "Accuracy Trap."

---

## 📊 Project Workflow & Features

The project walks through a rigorous machine learning workflow structured across distinct experimental steps:

1. **Exploratory Data Analysis (EDA):** Baseline evaluation of credit portfolios, capturing missing values (`?` markers), and analyzing descriptive distribution summaries.
2. **Target Variable Dissection:** Isolating features and highlighting a severe target variable skew (90% Paid vs. 10% Unpaid records).
3. **Mathematical Data Imputation:** Handling missing values gracefully by deploying a mean imputation approach for continuous features and mode imputation for categorical indices.
4. **Categorical Matrix Encoding:** Utilizing One-Hot Encoding (`drop_first=True`) to transition text values into mathematical arrays while actively avoiding the dummy variable trap.
5. **Stratified Splitting & Scaled Normalization:** Dividing the dataset into a stratified 70/30 train/test split to perfectly lock in the minority class distribution ratio, alongside scaling using `MinMaxScaler`.
6. **Hyperparameter Grid Search:** Evaluating hyperparameter combinations (`C` regularization penalties, `class_weight` tunings, and `tol` optimization limits) scored via a weighted $F_1$ formula.
7. **Visual Diagnostic Dashboards:** Generating inline Seaborn heatmaps for the Confusion Matrix alongside automated Receiver Operating Characteristic (ROC) curve computations.

---

## 📈 Model Performance Insights

Despite achieving a misleading **90.00% overall Accuracy Score**, the diagnostic visualizations reveal a key machine learning trap:

* **The Dummy Classifier Bias:** Because of the extreme data scarcity and skew (100 total rows), the optimized model maxes out accuracy by predicting the majority class (Fully Paid) across 100% of test scenarios.
* **Minority Metric Breakdown:** The true positive count for default predictions resulted in flat **0.00 Precision, Recall, and F1-scores**, pinpointing a massive operational risk bottleneck.

### Summary of Next Steps
To evolve this framework for true predictive capability, subsequent variations will incorporate **SMOTE (Synthetic Minority Over-sampling Technique)** and transition into non-linear ensemble frameworks like **Random Forests** or **XGBoost**.

---

## ⚙️ Requirements & Installation

Make sure you have Python installed along with the primary numerical and tracking packages.

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
