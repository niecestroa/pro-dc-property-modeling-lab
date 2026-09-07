# =============================================================================
# run_all.R — DC Property ML End-to-End Workflow
# =============================================================================
# Usage: Rscript run_all.R               (runs everything)
#        Rscript run_all.R --skip-data   (assumes data/splits already exist)
# =============================================================================

suppressPackageStartupMessages(library(here))

# ---- 0. Bootstrap: resolve project root via {here} -------------------------
cat("\n", strrep("=", 70), "\n")
cat("  DC Property ML — End-to-End Workflow\n")
cat("  Author: Aaron Niecestro\n")
cat(strrep("=", 70), "\n\n")

# ---- 1. Source all modules --------------------------------------------------
source(here("r_code", "01_load_config.R"))
source(here("r_code", "02_load_data.R"))
source(here("r_code", "03_split_data.R"))
source(here("r_code", "04_feature_engineering.R"))
source(here("r_code", "05_models_linear.R"))
source(here("r_code", "06_models_polynomial.R"))
source(here("r_code", "07_models_knn.R"))
source(here("r_code", "08_models_lda_qda.R"))
source(here("r_code", "09_models_ridge_lasso.R"))
source(here("r_code", "10_models_pcr_pls.R"))
source(here("r_code", "11_models_trees.R"))
source(here("r_code", "12_models_random_forest.R"))
source(here("r_code", "13_models_svm.R"))
source(here("r_code", "regression_metrics.R"))

source(here("utils", "utils_io.R"))
source(here("utils", "utils_plots.R"))
source(here("utils", "utils_preprocessing.R"))

# ---- 2. Load config ---------------------------------------------------------
cfg <- load_config()
set_project_seed(cfg)
ensure_output_dirs(cfg)

args <- commandArgs(trailingOnly = TRUE)
skip_data <- "--skip-data" %in% args

# ===========================================================================
# PHASE 1 — DATA
# ===========================================================================
cat("\n--- PHASE 1: Data Loading & Splitting ---\n")

if (!skip_data) {
  df    <- load_raw_data(cfg)
  save_processed_data(df, cfg)
  splits <- make_split(df, cfg)
} else {
  message("[run_all] --skip-data: loading pre-saved splits.")
  df     <- readRDS(here(cfg$paths$processed_data))
  splits <- load_splits(cfg)
}

train <- splits$train
test  <- splits$test

# RF-safe copies (critical fix)
train_rf <- train
test_rf  <- test

# Add classification labels
labeled <- add_classification_labels(train, test, cfg)
train_c <- labeled$train
test_c  <- labeled$test

# ===========================================================================
# PHASE 2 — EDA
# ===========================================================================
cat("\n--- PHASE 2: Exploratory Data Analysis ---\n")

cat("\nData Quality Report:\n")
data_quality_report(df)

cat("\nDescriptive Statistics:\n")
print(numeric_summary(df))

save_price_histogram(df, cfg)
save_geo_price_map(df, cfg)
save_correlation_heatmap(df, cfg)
for (cat_var in c("Grade", "Condition", "Ward")) {
  if (cat_var %in% colnames(df)) save_price_by_category(df, cat_var, cfg)
}

# ===========================================================================
# PHASE 3 — FEATURE ENGINEERING
# ===========================================================================
cat("\n--- PHASE 3: Feature Engineering ---\n")

train_poly <- add_polynomial_features(train, cfg)
test_poly  <- add_polynomial_features(test,  cfg)

train_Xy <- build_Xy(train_poly)
test_Xy  <- build_Xy(test_poly)

# ===========================================================================
# PHASE 4 — REGRESSION MODELS
# ===========================================================================
cat("\n--- PHASE 4: Regression Models ---\n")

## 4a. Linear Regression
cat("\n[4a] Linear Regression\n")
lm_model <- fit_linear(train)
save_model(lm_model, cfg, "linear")
save_residual_plots(lm_model, cfg, tag = "linear_diagnostics")
check_vif(lm_model)

lm_pred   <- predict(lm_model, newdata = test)
lm_metrics <- regression_metrics(test$Price_10K, lm_pred, "Linear")
log_result(cfg, "Linear", "Test_MSE", lm_metrics["MSE"])
save_pred_vs_actual(test$Price_10K, lm_pred, cfg, "Linear Regression")

## 4b. Polynomial Regression
cat("\n[4b] Polynomial Regression\n")
poly_model    <- fit_polynomial(train, cfg)
outlier_idx   <- detect_outliers(poly_model)
poly_refitted <- refit_without_outliers(train, cfg, outlier_idx)
poly_model    <- poly_refitted$model
save_model(poly_model, cfg, "polynomial")
save_diagnostic_plots(poly_model, cfg, tag = "polynomial_diagnostics")

poly_pred    <- predict(poly_model, newdata = test)
poly_metrics <- regression_metrics(test$Price_10K, poly_pred, "Polynomial")
log_result(cfg, "Polynomial", "Test_MSE", poly_metrics["MSE"])
save_pred_vs_actual(test$Price_10K, poly_pred, cfg, "Polynomial Regression")

## 4c. Ridge Regression
cat("\n[4c] Ridge Regression\n")
ridge_model <- fit_ridge(train_Xy, cfg)
save_model(ridge_model, cfg, "ridge")
save_cv_plot(ridge_model, cfg, "ridge_cv")
save_coef_path_plot(ridge_model, cfg, "ridge_coef_path")

ridge_eval <- eval_ridge(ridge_model, test_Xy)
log_result(cfg, "Ridge", "Test_MSE", ridge_eval$mse)

## 4d. Lasso Regression
cat("\n[4d] Lasso Regression\n")
lasso_model <- fit_lasso(train_Xy, cfg)
save_model(lasso_model, cfg, "lasso")
save_cv_plot(lasso_model, cfg, "lasso_cv")
save_coef_path_plot(lasso_model, cfg, "lasso_coef_path")

lasso_eval <- eval_lasso(lasso_model, test_Xy)
log_result(cfg, "Lasso", "Test_MSE", lasso_eval$mse)
compare_ridge_lasso(ridge_eval, lasso_eval)

## 4e. PCR
cat("\n[4e] Principal Component Regression (PCR)\n")
pcr_model <- fit_pcr(train_poly, cfg)
save_model(pcr_model, cfg, "pcr")
save_rmsep_plot(pcr_model, cfg, "pcr_rmsep")
save_variance_plot(pcr_model, cfg, "pcr_variance")

pcr_eval    <- eval_pcr(pcr_model, test_poly)
log_result(cfg, "PCR", "Test_MSE", pcr_eval$mse,
           notes = paste("ncomp =", pcr_eval$ncomp_used))

## 4f. PLS
cat("\n[4f] Partial Least Squares (PLS)\n")
pls_model <- fit_pls(train_poly, cfg)
save_model(pls_model, cfg, "pls")
save_rmsep_plot(pls_model, cfg, "pls_rmsep")
save_variance_plot(pls_model, cfg, "pls_variance")

pls_eval    <- eval_pls(pls_model, test_poly)
log_result(cfg, "PLS", "Test_MSE", pls_eval$mse,
           notes = paste("ncomp =", pls_eval$ncomp_used))

## 4g. Regression Tree + Pruning
cat("\n[4g] Regression Tree (with pruning)\n")
tree_model   <- fit_tree(train, cfg)
save_tree_plot(tree_model, cfg, "tree_unpruned")
save_cp_plot(tree_model, cfg)

pruned_tree  <- prune_tree(tree_model, cfg)
save_tree_plot(pruned_tree, cfg, "tree_pruned")
save_model(pruned_tree, cfg, "tree_pruned")

tree_eval    <- eval_tree(pruned_tree, test)
log_result(cfg, "Regression Tree (Pruned)", "Test_MSE", tree_eval$mse)

## 4h. Random Forest + Bagging
cat("\n[4h] Random Forest & Bagging\n")

# Default RF
rf_model    <- fit_random_forest(train_rf, cfg)
save_model(rf_model, cfg, "random_forest_full")
save_rf_error_plot(rf_model, cfg, "rf_oob_error_full")
save_importance_plot(rf_model, cfg, "rf_importance_full")

# Optimal RF
rf_optimal  <- fit_rf_optimal(train_rf, cfg)
save_model(rf_optimal, cfg, "random_forest_optimal")
save_importance_plot(rf_optimal, cfg, "rf_importance_optimal")

# Bagging
bag_model   <- fit_bagging(train_rf, cfg)
save_model(bag_model, cfg, "bagging")

# Unified RF evaluation
rf_eval     <- eval_rf(rf_optimal, test_rf)
log_result(cfg, "Random Forest (Optimal)", "Test_MSE", rf_eval$MSE)

save_pred_vs_actual(test_rf$Price_10K, rf_eval$pred, cfg, "Random Forest Optimal")

cat("\n--- PHASE 5: Classification Models ---\n")

# ============================================================
# MEAN THRESHOLD CLASSIFICATION
# ============================================================
cat("\n[5a] Mean Threshold Classification (class_mean)\n")

# KNN (mean)
knn_results_mean <- knn_loop(train_c, test_c, cfg = cfg, class_col = "class_mean")
plot_knn_error(knn_results_mean, cfg)
best_k_mean      <- knn_results_mean$k[which.min(knn_results_mean$mse)]
knn_best_mean    <- run_knn(train_c, test_c, k = best_k_mean, class_col = "class_mean")

# LDA (mean)
lda_model_mean <- fit_lda(train_c, "class_mean", priors = cfg$models$lda$priors_equal)
lda_eval_mean  <- eval_lda(lda_model_mean, test_c, "class_mean")

# QDA (mean)
qda_model_mean <- fit_qda(train_c, "class_mean", priors = cfg$models$qda$priors_equal)
qda_eval_mean  <- eval_qda(qda_model_mean, test_c, "class_mean")

# SVM (mean)
svm_kernel_results_mean <- compare_svm_kernels(train_c, test_c, "class_mean", cfg)
best_kernel_mean <- svm_kernel_results_mean$Kernel[
  which.min(svm_kernel_results_mean$Misclassification_Rate)
]
svm_best_mean <- fit_svm(train_c, kernel = best_kernel_mean,
                         class_col = "class_mean", cfg = cfg)
svm_eval_mean <- eval_svm(svm_best_mean, test_c, "class_mean")


# ============================================================
# MEDIAN THRESHOLD CLASSIFICATION
# ============================================================
cat("\n[5b] Median Threshold Classification (class_median)\n")

# KNN (median)
knn_results_median <- knn_loop(train_c, test_c, cfg = cfg, class_col = "class_median")
plot_knn_error(knn_results_median, cfg)
best_k_median      <- knn_results_median$k[which.min(knn_results_median$mse)]
knn_best_median    <- run_knn(train_c, test_c, k = best_k_median, class_col = "class_median")

# LDA (median)
lda_model_median <- fit_lda(train_c, "class_median", priors = cfg$models$lda$priors_equal)
lda_eval_median  <- eval_lda(lda_model_median, test_c, "class_median")

# QDA (median)
qda_model_median <- fit_qda(train_c, "class_median", priors = cfg$models$qda$priors_equal)
qda_eval_median  <- eval_qda(qda_model_median, test_c, "class_median")

# SVM (median)
svm_kernel_results_median <- compare_svm_kernels(train_c, test_c, "class_median", cfg)
best_kernel_median <- svm_kernel_results_median$Kernel[
  which.min(svm_kernel_results_median$Misclassification_Rate)
]
svm_best_median <- fit_svm(train_c, kernel = best_kernel_median,
                           class_col = "class_median", cfg = cfg)
svm_eval_median <- eval_svm(svm_best_median, test_c, "class_median")

cat("\n--- PHASE 6: Model Comparison Summary ---\n")

# classification_comparison_table() defined in utils/utils_plots.R1

# ============================================================
# MEAN THRESHOLD RESULTS TABLE
# ============================================================
cls_results_mean <- list(
  "KNN (Best k)"       = list(actual = test_c$class_mean, pred = knn_best_mean$pred),
  "LDA (equal priors)" = list(actual = test_c$class_mean, pred = lda_eval_mean$pred),
  "QDA (equal priors)" = list(actual = test_c$class_mean, pred = qda_eval_mean$pred),
  "SVM (Best Kernel)"  = list(actual = test_c$class_mean, pred = svm_eval_mean$pred)
)

cls_table_mean <- classification_comparison_table(cls_results_mean)
save_comparison_table(cls_table_mean, cfg, "classification_mean")
save_classification_bar_chart(cls_table_mean, cfg, "classification_mean")
saveRDS(cls_table_mean, here(cfg$paths$model_dir, "classification_mean_table.rds"))

save_confusion_matrix(test_c$class_mean, knn_best_mean$pred, cfg, "mean_knn")


# ============================================================
# MEDIAN THRESHOLD RESULTS TABLE
# ============================================================
cls_results_median <- list(
  "KNN (Best k)"       = list(actual = test_c$class_median, pred = knn_best_median$pred),
  "LDA (equal priors)" = list(actual = test_c$class_median, pred = lda_eval_median$pred),
  "QDA (equal priors)" = list(actual = test_c$class_median, pred = qda_eval_median$pred),
  "SVM (Best Kernel)"  = list(actual = test_c$class_median, pred = svm_eval_median$pred)
)

cls_table_median <- classification_comparison_table(cls_results_median)
save_comparison_table(cls_table_median, cfg, "classification_median")
save_classification_bar_chart(cls_table_median, cfg, "classification_median")
saveRDS(cls_table_median, here(cfg$paths$model_dir, "classification_median_table.rds"))

save_confusion_matrix(test_c$class_median, knn_best_median$pred, cfg, "median_knn")


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================
cat("\n--- Mean Threshold Classification ---\n")
print(cls_table_mean)

cat("\n--- Median Threshold Classification ---\n")
print(cls_table_median)

cat("\n", strrep("=", 70), "\n")
cat("  Run complete. All outputs in outputs/\n")
cat(strrep("=", 70), "\n\n")
