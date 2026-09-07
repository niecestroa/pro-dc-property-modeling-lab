# =============================================================================
# 13_models_svm.R
# Support Vector Machines for binary classification of Price_10K.
# Adds hyperparameter tuning (cost × gamma grid search).
# =============================================================================

suppressPackageStartupMessages({
  library(e1071)
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------------
# Helper: numeric predictor matrix for SVM
# ---------------------------------------------------------------------------
.svm_Xy <- function(df, class_col = "class_median") {
  exclude <- c("Price_10K", "class_mean", "class_median")
  X <- df %>%
    dplyr::select(-dplyr::all_of(exclude)) %>%
    dplyr::select(where(is.numeric))
  y <- df[[class_col]]
  list(X = X, y = y)
}

# ---------------------------------------------------------------------------
# Single-kernel SVM (default)
# ---------------------------------------------------------------------------

fit_svm <- function(train, kernel = "radial", class_col = "class_median", cfg) {
  cost  <- cfg$models$svm$cost_default
  gamma <- cfg$models$svm$gamma_default
  
  xy <- .svm_Xy(train, class_col)
  
  message(sprintf("[fit_svm] Fitting SVM | kernel=%s | cost=%.4f | gamma=%.4f",
                  kernel, cost, gamma))
  
  model <- e1071::svm(
    x      = xy$X,
    y      = xy$y,
    kernel = kernel,
    cost   = cost,
    gamma  = gamma,
    type   = "C-classification",
    scale  = TRUE
  )
  
  message(sprintf("[fit_svm] Number of support vectors: %d", sum(model$nSV)))
  model
}

# ---------------------------------------------------------------------------
# Evaluate SVM (default)
# ---------------------------------------------------------------------------

eval_svm <- function(model, test, class_col = "class_median") {
  xy   <- .svm_Xy(test, class_col)
  pred <- predict(model, newdata = xy$X)
  mse  <- mean(pred != xy$y)
  conf <- table(Predicted = pred, Actual = xy$y)
  message(sprintf("[eval_svm] Misclassification rate = %.7f", mse))
  list(pred = pred, mse = mse, confusion = conf)
}

# ---------------------------------------------------------------------------
# Multi-kernel sweep (default models)
# ---------------------------------------------------------------------------

compare_svm_kernels <- function(train, test, class_col = "class_median", cfg) {
  kernels <- cfg$models$svm$kernels
  
  results <- purrr::map_dfr(kernels, function(k) {
    m <- fit_svm(train, kernel = k, class_col = class_col, cfg = cfg)
    e <- eval_svm(m, test, class_col = class_col)
    tibble::tibble(
      Kernel                 = k,
      Misclassification_Rate = e$mse,
      Correct_Rate           = 1 - e$mse
    )
  })
  
  message("\n[compare_svm_kernels] Kernel comparison:")
  print(results)
  invisible(results)
}

# ---------------------------------------------------------------------------
# Hyperparameter tuning (cost × gamma grid search)
# ---------------------------------------------------------------------------

svm_tune <- function(train, class_col = "class_median", cfg) {
  
  xy <- .svm_Xy(train, class_col)
  
  cost_values  <- cfg$models$svm$cost_grid
  gamma_values <- cfg$models$svm$gamma_grid
  
  message("[svm_tune] Starting grid search...")
  
  tune_result <- e1071::tune(
    method  = e1071::svm,
    train.x = xy$X,
    train.y = xy$y,
    kernel  = cfg$models$svm$kernel_default,
    ranges  = list(cost = cost_values, gamma = gamma_values),
    type    = "C-classification",
    scale   = TRUE
  )
  
  best <- tune_result$best.parameters
  message(sprintf("[svm_tune] Best cost=%.4f | Best gamma=%.4f",
                  best$cost, best$gamma))
  
  final_model <- e1071::svm(
    x      = xy$X,
    y      = xy$y,
    kernel = cfg$models$svm$kernel_default,
    cost   = best$cost,
    gamma  = best$gamma,
    type   = "C-classification",
    scale  = TRUE
  )
  
  list(
    model = final_model,
    best  = best,
    tune  = tune_result
  )
}

# ---------------------------------------------------------------------------
# Evaluate tuned SVM
# ---------------------------------------------------------------------------

eval_svm_tuned <- function(tuned, test, class_col = "class_median") {
  xy   <- .svm_Xy(test, class_col)
  pred <- predict(tuned$model, newdata = xy$X)
  mse  <- mean(pred != xy$y)
  conf <- table(Predicted = pred, Actual = xy$y)
  
  message(sprintf("[eval_svm_tuned] Misclassification rate = %.7f", mse))
  list(pred = pred, mse = mse, confusion = conf, best = tuned$best)
}

# ---------------------------------------------------------------------------
# Compare default vs tuned SVM
# ---------------------------------------------------------------------------

compare_svm_tuned_vs_default <- function(train, test, class_col = "class_median", cfg) {
  
  default_model <- fit_svm(train, kernel = cfg$models$svm$kernel_default,
                           class_col = class_col, cfg = cfg)
  default_eval  <- eval_svm(default_model, test, class_col)
  
  tuned         <- svm_tune(train, class_col, cfg)
  tuned_eval    <- eval_svm_tuned(tuned, test, class_col)
  
  tibble::tibble(
    Model                 = c("Default SVM", "Tuned SVM"),
    Misclassification_Rate = c(default_eval$mse, tuned_eval$mse),
    Correct_Rate           = 1 - c(default_eval$mse, tuned_eval$mse),
    Cost                   = c(cfg$models$svm$cost_default, tuned$best$cost),
    Gamma                  = c(cfg$models$svm$gamma_default, tuned$best$gamma)
  )
}

# ---------------------------------------------------------------------------
# Scatter plot (Lat × Lon)
# ---------------------------------------------------------------------------

save_svm_scatter <- function(df, class_col = "class_median", cfg, tag = "svm_scatter") {
  
  # FIX: convert numeric class labels (0/1) to factor BEFORE plotting
  df[[class_col]] <- factor(df[[class_col]])
  
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  
  p <- ggplot2::ggplot(
    df %>% dplyr::sample_n(min(5000, nrow(df))),
    ggplot2::aes(x = Longitude, y = Latitude, colour = .data[[class_col]])
  ) +
    ggplot2::geom_point(alpha = 0.4, size = 0.8) +
    ggplot2::scale_colour_manual(values = c("steelblue", "firebrick")) +
    ggplot2::labs(
      title  = "DC Properties — Price Classification (Lat × Lon)",
      colour = class_col
    ) +
    ggplot2::theme_bw()
  
  ggplot2::ggsave(out, p, width = 8, height = 6, dpi = 150)
  message(sprintf("[save_svm_scatter] Saved → %s", out))
}

