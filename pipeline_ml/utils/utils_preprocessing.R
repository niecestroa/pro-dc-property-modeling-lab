# =============================================================================
# utils/utils_preprocessing.R
# Reusable data-quality and summary helpers used across EDA and modelling.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

#' Print a tidy summary of all columns: type, N missing, % missing, N unique
#'
#' @param df  Any tibble or data.frame.
#' @return Tibble of column-level diagnostics (printed + returned invisibly).
data_quality_report <- function(df) {
  n <- nrow(df)
  report <- purrr::map_dfr(colnames(df), function(col) {
    x <- df[[col]]
    tibble::tibble(
      Column    = col,
      Type      = class(x)[1],
      N_Missing = sum(is.na(x)),
      Pct_Miss  = round(100 * sum(is.na(x)) / n, 2),
      N_Unique  = dplyr::n_distinct(x)
    )
  })
  cat("\n=== Data Quality Report ===\n")
  print(report, n = Inf)
  cat(sprintf("\nTotal rows: %d | Total cols: %d\n\n", n, ncol(df)))
  invisible(report)
}

#' Summarise numeric predictors: mean, sd, min, median, max
#'
#' Reproduces Table 1 of the paper (descriptive statistics).
#'
#' @param df Cleaned tibble.
#' @return Tibble with one row per numeric column.
numeric_summary <- function(df) {
  df %>%
    dplyr::select(where(is.numeric), -dplyr::starts_with("class_")) %>%
    tidyr::pivot_longer(dplyr::everything(), names_to = "Variable", values_to = "Value") %>%
    dplyr::group_by(Variable) %>%
    dplyr::summarise(
      Mean   = round(mean(Value,   na.rm = TRUE), 4),
      SD     = round(sd(Value,     na.rm = TRUE), 4),
      Min    = round(min(Value,    na.rm = TRUE), 4),
      Median = round(median(Value, na.rm = TRUE), 4),
      Max    = round(max(Value,    na.rm = TRUE), 4),
      .groups = "drop"
    ) %>%
    dplyr::arrange(Variable)
}

#' Frequency table for a categorical variable
#'
#' @param df  Tibble.
#' @param col Column name (character).
#' @return Tibble with columns: Level, N, Pct.
freq_table <- function(df, col) {
  df %>%
    dplyr::count(.data[[col]], name = "N") %>%
    dplyr::mutate(Pct = round(100 * N / sum(N), 2)) %>%
    dplyr::rename(Level = .data[[col]]) %>%
    dplyr::arrange(dplyr::desc(N))
}

#' Scale / center numeric columns (min-max or z-score)
#'
#' @param train    Training tibble.
#' @param test     Testing tibble.
#' @param cols     Columns to scale. Defaults to all numeric predictors.
#' @param method   "zscore" (default) or "minmax".
#' @return List $train and $test with scaled columns; $params stores scale factors.
scale_splits <- function(train, test, cols = NULL, method = "zscore") {
  if (is.null(cols)) {
    exclude <- c("Price_10K", "class_mean", "class_median")
    cols    <- train %>%
      dplyr::select(where(is.numeric), -dplyr::all_of(exclude)) %>%
      colnames()
  }
  
  params <- list()
  for (col in cols) {
    if (method == "zscore") {
      mu           <- mean(train[[col]], na.rm = TRUE)
      sigma        <- sd(train[[col]],   na.rm = TRUE)
      params[[col]] <- list(mu = mu, sigma = sigma)
      train[[col]]  <- (train[[col]] - mu) / sigma
      test[[col]]   <- (test[[col]]  - mu) / sigma
    } else {
      mn           <- min(train[[col]], na.rm = TRUE)
      mx           <- max(train[[col]], na.rm = TRUE)
      params[[col]] <- list(min = mn, max = mx)
      train[[col]]  <- (train[[col]] - mn) / (mx - mn)
      test[[col]]   <- (test[[col]]  - mn) / (mx - mn)
    }
  }
  
  message(sprintf("[scale_splits] Scaled %d columns using '%s' (fit on train only).",
                  length(cols), method))
  list(train = train, test = test, params = params)
}

add_classification_labels <- function(train, test, cfg) {
  
  # Compute quartile thresholds from TRAIN ONLY (avoid leakage)
  qs <- quantile(train$Price_10K, probs = c(0, 0.25, 0.50, 0.75, 1), na.rm = TRUE)
  
  # Create 4-class factor based on quartiles
  train$class_price4 <- cut(
    train$Price_10K,
    breaks = qs,
    include.lowest = TRUE,
    labels = c("VeryLow", "Low", "High", "VeryHigh")
  )
  
  test$class_price4 <- cut(
    test$Price_10K,
    breaks = qs,
    include.lowest = TRUE,
    labels = c("VeryLow", "Low", "High", "VeryHigh")
  )
  
  # Also create 3-class version (optional)
  train$class_price3 <- cut(
    train$Price_10K,
    breaks = quantile(train$Price_10K, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
    include.lowest = TRUE,
    labels = c("Low", "Medium", "High")
  )
  
  test$class_price3 <- cut(
    test$Price_10K,
    breaks = quantile(train$Price_10K, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
    include.lowest = TRUE,
    labels = c("Low", "Medium", "High")
  )
  
  # Keep binary labels too (for backward compatibility)
  mean_price   <- mean(train$Price_10K, na.rm = TRUE)
  median_price <- median(train$Price_10K, na.rm = TRUE)
  
  train$class_mean   <- ifelse(train$Price_10K >= mean_price,   1, 0)
  train$class_median <- ifelse(train$Price_10K >= median_price, 1, 0)
  
  test$class_mean    <- ifelse(test$Price_10K >= mean_price,   1, 0)
  test$class_median  <- ifelse(test$Price_10K >= median_price, 1, 0)
  
  list(train = train, test = test)
}
