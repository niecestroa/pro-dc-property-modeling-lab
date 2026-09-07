# =============================================================================
# 07_models_knn.R
# K-Nearest Neighbor classification.
# Binary target:  class_mean   ("Reasonable" / "Overpriced", threshold 57.725)
# Supports:       fixed-k runs, k-loop (1–50), and 3-class variant.
# =============================================================================

suppressPackageStartupMessages({
  library(class)    # knn()
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#' Select numeric-only predictors for KNN (KNN requires numeric input)
#'
#' @param df Tibble.
#' @return Tibble with only numeric predictor columns (no response/class cols).
.knn_predictors <- function(df) {
  exclude <- c("Price_10K", "class_mean", "class_median")
  num_cols <- df %>%
    dplyr::select(-dplyr::all_of(exclude)) %>%
    dplyr::select(where(is.numeric)) %>%
    colnames()
  df[, num_cols, drop = FALSE]
}

# ---------------------------------------------------------------------------
# Single-k run
# ---------------------------------------------------------------------------

#' Fit and evaluate KNN for one value of k
#'
#' @param train      Training tibble (with class labels from add_classification_labels()).
#' @param test       Testing tibble (with class labels).
#' @param k          Number of neighbors.
#' @param class_col  Name of the classification column. Default "class_mean".
#' @return List: $pred (factor), $mse (misclassification rate), $k.
run_knn <- function(train, test, k, class_col = "class_mean") {
  X_train <- .knn_predictors(train)
  X_test  <- .knn_predictors(test)
  y_train <- train[[class_col]]
  y_test  <- test[[class_col]]

  pred <- class::knn(
    train = X_train,
    test  = X_test,
    cl    = y_train,
    k     = k
  )

  mse <- mean(pred != y_test)
  message(sprintf("[run_knn] k = %3d | Misclassification rate = %.6f", k, mse))
  list(pred = pred, mse = mse, k = k)
}

# ---------------------------------------------------------------------------
# K-loop: sweep k = 1 to k_max
# ---------------------------------------------------------------------------

#' Run KNN for k = 1 to k_max and return a results table
#'
#' @param train     Training tibble.
#' @param test      Testing tibble.
#' @param k_max     Maximum k to evaluate. Sourced from cfg$models$knn$k_loop_max.
#' @param class_col Classification column name.
#' @param cfg       Config list (used to source k_max if not supplied).
#' @return Tibble with columns k and mse, sorted by k.
knn_loop <- function(train, test, k_max = NULL, class_col = "class_mean", cfg = NULL) {
  # Determine k_max from config if not supplied
  if (is.null(k_max) && !is.null(cfg)) k_max <- cfg$models$knn$k_loop_max
  if (is.null(k_max)) k_max <- 50L
  
  message(sprintf("[knn_loop] Sweeping k = 1 to %d ...", k_max))
  
  X_train <- .knn_predictors(train)
  X_test  <- .knn_predictors(test)
  y_train <- train[[class_col]]
  y_test  <- test[[class_col]]
  
  # --- Progress bar ---------------------------------------------------------
  pb <- txtProgressBar(min = 1, max = k_max, style = 3)
  
  results <- purrr::map_dfr(seq_len(k_max), function(k) {
    pred <- class::knn(train = X_train, test = X_test, cl = y_train, k = k)
    mse  <- mean(pred != y_test)
    
    setTxtProgressBar(pb, k)   # update progress bar
    
    tibble::tibble(k = k, mse = mse)
  })
  
  close(pb)
  # --------------------------------------------------------------------------
  
  best <- results[which.min(results$mse), ]
  message(sprintf("[knn_loop] Best k = %d | MSE = %.6f", best$k, best$mse))
  
  results
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

#' Plot misclassification rate vs. k
#'
#' @param knn_results Tibble from knn_loop().
#' @param cfg         Config list.
#' @param save        Save plot to file? Default TRUE.
plot_knn_error <- function(knn_results, cfg = NULL, save = TRUE) {
  best_k   <- knn_results$k[which.min(knn_results$mse)]
  best_mse <- min(knn_results$mse)

  p <- ggplot2::ggplot(knn_results, ggplot2::aes(x = k, y = mse)) +
    ggplot2::geom_line(color = "steelblue", linewidth = 0.8) +
    ggplot2::geom_point(color = "steelblue", size = 1) +
    ggplot2::geom_vline(xintercept = best_k, linetype = "dashed", color = "firebrick") +
    ggplot2::annotate("text", x = best_k + 2, y = best_mse,
                      label = sprintf("Best k=%d\nMSE=%.4f", best_k, best_mse),
                      hjust = 0, size = 3.5, color = "firebrick") +
    ggplot2::labs(title = "KNN: Misclassification Rate vs. k",
                  x = "k (Number of Neighbors)",
                  y = "Misclassification Rate") +
    ggplot2::theme_bw()

  if (save && !is.null(cfg)) {
    out <- file.path(here::here(cfg$paths$plot_dir), "knn_error_vs_k.png")
    ggplot2::ggsave(out, p, width = 8, height = 5, dpi = 150)
    message(sprintf("[plot_knn_error] Saved → %s", out))
  }
  invisible(p)
}
