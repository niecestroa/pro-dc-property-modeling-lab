# =============================================================================
# 14_evaluate_report.R
# Unified Model Comparison Report (RMarkdown-ready)
# + Residual Diagnostics (Histogram, QQ, Residual vs Fitted)
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

# ===========================================================================
# Classification Comparison Table (multi-class safe)
# ===========================================================================
classification_comparison_table <- function(results) {
  tibble::tibble(
    Model    = names(results),
    Accuracy = purrr::map_dbl(results, ~ mean(.x$pred == .x$actual)),
    Classes  = purrr::map_chr(results, ~ paste(levels(.x$actual), collapse = ", "))
  )
}

compute_f1 <- function(actual, pred) {
  cm <- table(actual, pred)
  precision <- diag(cm) / colSums(cm)
  recall    <- diag(cm) / rowSums(cm)
  f1        <- 2 * precision * recall / (precision + recall)
  macro_f1  <- mean(f1, na.rm = TRUE)
  weighted_f1 <- sum(f1 * rowSums(cm)) / sum(cm)
  list(macro_f1 = macro_f1, weighted_f1 = weighted_f1)
}

classification_comparison_table <- function(results) {
  tibble::tibble(
    Model       = names(results),
    Accuracy    = purrr::map_dbl(results, ~ mean(.x$pred == .x$actual)),
    Macro_F1    = purrr::map_dbl(results, ~ compute_f1(.x$actual, .x$pred)$macro_f1),
    Weighted_F1 = purrr::map_dbl(results, ~ compute_f1(.x$actual, .x$pred)$weighted_f1),
    Classes     = purrr::map_chr(results, ~ paste(levels(.x$actual), collapse = ", "))
  )
}

# ===========================================================================
# Helper: Ranger Prediction Wrapper
# ===========================================================================
eval_ranger <- function(model, test) {
  pred <- predict(model, data = test)$predictions
  list(
    actual = test$Price_10K,
    pred   = pred
  )
}

# ===========================================================================
# Residual Diagnostics
# ===========================================================================
save_residual_plots <- function(actual, pred, cfg, label = "model") {
  
  residuals <- actual - pred
  df <- data.frame(
    Actual     = actual,
    Predicted  = pred,
    Residuals  = residuals
  )
  
  # -----------------------------
  # 1. Residual Histogram
  # -----------------------------
  out_hist <- file.path(
    here::here(cfg$paths$plot_dir),
    paste0("residual_hist_", gsub(" ", "_", label), ".png")
  )
  
  p_hist <- ggplot(df, aes(x = Residuals)) +
    geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
    labs(title = paste(label, "— Residual Histogram"),
         x = "Residuals", y = "Count") +
    theme_bw()
  
  ggsave(out_hist, p_hist, width = 7, height = 5, dpi = 150)
  message(sprintf("[save_residual_plots] Saved histogram → %s", out_hist))
  
  # -----------------------------
  # 2. QQ Plot
  # -----------------------------
  out_qq <- file.path(
    here::here(cfg$paths$plot_dir),
    paste0("residual_qq_", gsub(" ", "_", label), ".png")
  )
  
  p_qq <- ggplot(df, aes(sample = Residuals)) +
    stat_qq(color = "steelblue") +
    stat_qq_line(color = "firebrick") +
    labs(title = paste(label, "— Residual QQ Plot")) +
    theme_bw()
  
  ggsave(out_qq, p_qq, width = 7, height = 5, dpi = 150)
  message(sprintf("[save_residual_plots] Saved QQ plot → %s", out_qq))
  
  # -----------------------------
  # 3. Residual vs Fitted
  # -----------------------------
  out_rvf <- file.path(
    here::here(cfg$paths$plot_dir),
    paste0("residual_vs_fitted_", gsub(" ", "_", label), ".png")
  )
  
  p_rvf <- ggplot(df, aes(x = Predicted, y = Residuals)) +
    geom_point(alpha = 0.4, color = "steelblue") +
    geom_hline(yintercept = 0, color = "firebrick", linetype = "dashed") +
    labs(title = paste(label, "— Residuals vs Fitted"),
         x = "Predicted", y = "Residuals") +
    theme_bw()
  
  ggsave(out_rvf, p_rvf, width = 7, height = 5, dpi = 150)
  message(sprintf("[save_residual_plots] Saved residual vs fitted → %s", out_rvf))
  
  invisible(list(hist = p_hist, qq = p_qq, rvf = p_rvf))
}

# ===========================================================================
# Unified Regression Comparison (RF, Optimal RF, Bagging)
# ===========================================================================
compare_regression_models <- function(models, test, cfg) {
  
  results <- list()
  
  # -----------------------------
  # Evaluate each model
  # -----------------------------
  if (!is.null(models$rf_full)) {
    results$RF_Full <- eval_ranger(models$rf_full, test)
  }
  
  if (!is.null(models$rf_optimal)) {
    results$RF_Optimal <- eval_ranger(models$rf_optimal, test)
  }
  
  if (!is.null(models$rf_bagging)) {
    results$Bagging <- eval_ranger(models$rf_bagging, test)
  }
  
  # -----------------------------
  # Build comparison table
  # -----------------------------
  tbl <- regression_comparison_table(results)
  
  # Save CSV + bar chart
  save_comparison_table(tbl, cfg, tag = "regression_comparison")
  save_mse_bar_chart(tbl, cfg)
  
  # -----------------------------
  # Save residual diagnostics
  # -----------------------------
  purrr::walk2(results, names(results), function(res, label) {
    save_residual_plots(
      actual = res$actual,
      pred   = res$pred,
      cfg    = cfg,
      label  = label
    )
  })
  
  tbl
}

cat("\n=== Regression Models ===\n")
print(reg_table)

cat("\n=== Binary Classification ===\n")
print(cls_table)

cat("\n=== 3-Class Classification ===\n")
print(cls_table_3)

cat("\n=== 4-Class Classification ===\n")
print(cls_table_4)

# ===========================================================================
# RMarkdown Template Snippet
# ===========================================================================
# Paste this into your RMarkdown report:
#
# ```{r model-comparison}
# models <- list(
#   rf_full    = rf_full,
#   rf_optimal = rf_optimal,
#   rf_bagging = rf_bagging
# )
#
# comparison_tbl <- compare_regression_models(models, test, cfg)
# knitr::kable(comparison_tbl, caption = "Regression Model Comparison")
# ```
#
# Residual plots will be saved automatically and can be included with:
#
# ```{r residual-plots, echo=FALSE}
# knitr::include_graphics("outputs/plots/residual_hist_RF_Full.png")
# knitr::include_graphics("outputs/plots/residual_qq_RF_Full.png")
# knitr::include_graphics("outputs/plots/residual_vs_fitted_RF_Full.png")
# ```
