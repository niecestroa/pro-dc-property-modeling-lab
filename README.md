# **DC Property Modeling Lab**

A unified analytical system for modeling Washington, D.C. real estate data using rigorous statistical methods, Bayesian inference, classical GLMs, and modern machine‑learning engineering practices. The **DC Property Modeling Lab** provides a fully reproducible, modular workflow designed for professional‑grade modeling, diagnostics, and reporting.

The repository now includes **both regression and classification pipelines**, enabling continuous price prediction and price‑tier segmentation within the same end‑to‑end workflow.

The system integrates:

- Classical statistical modeling (GLM, inference, diagnostics, logistic regression)  
- Bayesian hierarchical modeling (Stan, JAGS)  
- Machine‑learning regression pipelines (trees, ensembles, regularization)  
- Machine‑learning classification pipelines (binary: mean/median split)  
- Modular R workflows with optional Python equivalents  
- Automated evaluation, visualization, and reporting  
- A structured ML manual documenting pipeline architecture and modeling methodology  

The goal is to demonstrate **end‑to‑end analytical engineering**, **transparent modeling**, and **reproducible computation** in a single cohesive system.

---

## **Repository Structure**

The project is organized into four major modeling modules:

```
bayesian/
glm/
pipeline_ml/
standalone_ml/
```

### **bayesian/**
Bayesian hierarchical modeling using Stan and JAGS, including:

- Hierarchical model specification  
- MCMC sampling  
- Posterior inference  
- Convergence diagnostics  
- Posterior predictive checks  

### **glm/**
Classical statistical modeling:

- **Linear regression**  
- **Logistic regression** (binary classification)  
- Regularized regression (ridge, lasso, elastic net)  
- Likelihood‑based inference  
- Diagnostics and interpretability workflows  

GLM models serve as interpretable baselines for both regression and classification tasks.

### **machine_learning_regression/**
Modern ML regression pipelines:

- Decision trees  
- Random forest  
- Bagging  
- PCR / PLS  
- Unified evaluation (MSE, RMSE, MAE, R²)  
- Residual diagnostics and predicted‑vs‑actual plots  

### **pipeline/**
The **end‑to‑end modeling pipeline**, integrating:

- Data loading and preprocessing  
- Feature engineering  
- Train/test splitting  
- Regression models  
- Classification models (binary mean/median)  
- Unified evaluation framework  
- Automated visualization  
- Saved comparison tables  
- Confusion matrices  
- Final reporting (RMarkdown)

This is the operational workflow that runs the entire modeling system via:

---

## **Classification Pipeline**

The modeling lab now includes full support for **price‑tier classification**, enabling segmentation tasks such as affordability analysis, market tiering, and risk scoring.

### **Supported Classification Tasks**
- **Binary classification**  
  Expensive vs Non‑Expensive (mean/median split)

### **Implemented Algorithms**
- Logistic regression (GLM module)  
- KNN  
- LDA  
- QDA  
- SVM (multiple kernels)

### **Evaluation & Visualization**
- Accuracy  
- Macro‑F1  
- Weighted‑F1  
- Confusion matrices  
- Multi‑class bar charts  
- Unified comparison tables  

Classification is fully integrated into the main pipeline and final report.

---

## **Key Features**

### **1. End‑to‑End Modeling Pipeline**
A complete, configurable workflow for DC property modeling:

- Data cleaning and preprocessing  
- Feature engineering  
- Train/test splitting  
- Regression and classification models  
- Unified evaluation metrics  
- Automated plots, diagnostics, and comparison tables  
- Caching for fast reproducible runs  

All components are controlled through a central `config.yaml`.

---

### **2. Bayesian Modeling**
Bayesian workflows implemented in **Stan** and **JAGS**, including:

- Hierarchical model specification  
- Posterior inference and uncertainty quantification  
- MCMC diagnostics  
- Posterior predictive validation  

These models provide probabilistic insight complementary to classical and ML approaches.

---

### **3. Generalized Linear Models (GLM)**
Classical statistical modeling using:

- Linear regression  
- **Logistic regression** (binary classification)  
- Regularization (ridge, lasso, elastic net)  
- Cross‑validation  
- Residual diagnostics  

GLMs serve as interpretable baselines for both regression and classification tasks.

---

### **4. Machine‑Learning Regression & Classification**
Modern ML pipelines including:

- Decision trees  
- Random forest  
- Bagging  
- Regularized regression  
- KNN, LDA, QDA, SVM classification  
- Unified evaluation and visualization  

Regression and classification now coexist within the same pipeline.

---

### **5. Machine Learning Manual**
A structured reference documenting:

- Modeling assumptions  
- Feature engineering strategies  
- Hyperparameter selection  
- Evaluation metrics  
- Pipeline architecture  
- Best practices for reproducible ML  

---

## **Technical Capabilities Demonstrated**

| Area | Capabilities | Tools |
|------|--------------|-------|
| **Statistical Modeling** | GLMs, logistic regression, inference, diagnostics | R, Python |
| **Bayesian Analysis** | Hierarchical models, MCMC | Stan, JAGS |
| **Machine Learning (Regression & Classification)** | Trees, ensembles, regularization, KNN, LDA/QDA, SVM | R, Python |
| **Pipeline Engineering** | Modular design, caching, reproducibility | R Projects, Git |
| **Visualization** | EDA, diagnostics, comparison charts | ggplot2, Python |
| **Documentation** | ML manual, structured reporting | Quarto, R Markdown |

---

## **Companion Repository**

For earlier academic case studies and foundational modeling work, see:

**DC Property Academic Statistical Analysis Portfolio**  
[DC Property Academic Version](https://github.com/niecestroa/dc-property-analysis-academic/tree/main)

---

## **Purpose and Vision**

The **DC Property Modeling Lab** is designed as a professional modeling environment that demonstrates:

- Reproducible analytical engineering  
- Transparent statistical and Bayesian modeling  
- Modern ML regression and classification workflows  
- Clean documentation and communication of modeling decisions  

It serves as a showcase of rigorous modeling practice suitable for data science, biostatistics, and applied analytics roles.
