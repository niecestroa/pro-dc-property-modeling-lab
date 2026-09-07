# **DC Property ML — Regression & Classification Models**

This folder contains the supervised learning components of the **DC Property ML** project — a full end‑to‑end machine‑learning pipeline for modeling Washington, D.C. residential property values. The system supports both **continuous price prediction** and **price‑tier classification**, enabling flexible analytics for valuation, segmentation, and risk assessment.

The module integrates seamlessly with the project’s preprocessing, feature engineering, evaluation, caching, and reporting framework.

---

## **Included Regression Models**

The following continuous‑outcome models are implemented:

- **Linear Regression**
- **Polynomial Regression**
- **Ridge Regression**
- **Lasso Regression**
- **PCR** — Principal Components Regression  
- **PLS** — Partial Least Squares  
- **Regression Trees**
- **Bagging**
- **Random Forest Regression**

Each model follows a unified interface and plugs directly into the project’s evaluation pipeline (MSE, RMSE, MAE, R², residual diagnostics, predicted‑vs‑actual plots).

---

## **Included Classification Models**

The project now includes full support for **price‑tier classification**, enabling segmentation tasks such as affordability analysis, market tiering, and risk scoring.

### **Binary Classification**
- Expensive vs Non‑Expensive (mean/median split)

### **3‑Class Classification**
- Low / Medium / High price tiers

### **4‑Class Classification**
- Quartiles of the price distribution

### **Implemented Algorithms**
- **KNN**
- **LDA**
- **QDA**
- **SVM (multiple kernels)**

All classification models integrate with:

- Accuracy, Macro‑F1, Weighted‑F1  
- Confusion matrices  
- Multi‑class bar charts  
- Unified comparison tables  
- Saved artifacts for reporting  

---

## **Pipeline Integration**

All models — regression and classification — plug into the main workflow:

- Data loading and preprocessing  
- Feature engineering (polynomial features, glmnet matrices)  
- Train/test split  
- Model training with caching  
- Unified evaluation framework  
- Automated visualization  
- Exported comparison tables  
- Saved artifacts for downstream reporting  

The module is fully compatible with:

- `run_all.R`  
- `config.yaml`  
- `utils/` (plots, IO, preprocessing)  
- The unified report generator (`14_evaluate_report.R`)  
- The RMarkdown report (`report.Rmd`)  

---

## **Purpose**

This folder documents the supervised learning components of the DC Property ML project. It ensures the codebase remains:

- **modular** — each model isolated and easy to maintain  
- **reproducible** — deterministic training and saved artifacts  
- **extensible** — new models can be added without breaking the pipeline  
- **comparable** — unified metrics and visualizations across all models  

The regression and classification modules together form the analytical backbone of the project’s valuation and segmentation capabilities.
