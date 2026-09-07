# =============================================================================
# 03_split_data.R
# Unified 50/50 train/test split — single source of truth for all models.
# The split is performed once and saved so every script uses identical data.
# =============================================================================

suppressPackageStartupMessages(library(dplyr))

#' Create and save a reproducible 50/50 train/test split
#'
#' @param df   Cleaned tibble from load_raw_data().
#' @param cfg  Config list from load_config().
#' @return Invisible list with elements $train and $test.
make_split <- function(df, cfg) {
  set.seed(cfg$project$seed)

  n      <- nrow(df)
  ratio  <- cfg$split$ratio                         # 0.50
  n_train <- floor(ratio * n)

  train_idx <- sample(seq_len(n), size = n_train, replace = FALSE)

  train <- df[ train_idx, ]
  test  <- df[-train_idx, ]

  message(sprintf("[make_split] Train: %d rows | Test: %d rows (%.0f%% / %.0f%%)",
                  nrow(train), nrow(test),
                  100 * nrow(train) / n,
                  100 * nrow(test)  / n))

  # Persist splits
  train_path <- here::here(cfg$paths$train_data)
  test_path  <- here::here(cfg$paths$test_data)

  dir.create(dirname(train_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(train, train_path)
  saveRDS(test,  test_path)

  message(sprintf("[make_split] Saved splits to %s", dirname(train_path)))
  invisible(list(train = train, test = test))
}

#' Load pre-saved splits from disk
#'
#' @param cfg Config list.
#' @return List with $train and $test tibbles.
load_splits <- function(cfg) {
  train_path <- here::here(cfg$paths$train_data)
  test_path  <- here::here(cfg$paths$test_data)

  if (!file.exists(train_path) || !file.exists(test_path)) {
    stop("[load_splits] Splits not found. Run make_split() first (or run_all.R).")
  }

  list(
    train = readRDS(train_path),
    test  = readRDS(test_path)
  )
}

#' Attach classification labels to a split for KNN / LDA / QDA / SVM
#'
#' Creates two new factor columns:
#'   class_mean   — "Reasonable" / "Overpriced"  (threshold = training mean)
#'   class_median — "Under"      / "Over"         (threshold = training median)
#'
#' @param train Training tibble.
#' @param test  Testing tibble.
#' @param cfg   Config list.
#' @return List $train and $test each with new class columns appended.
add_classification_labels <- function(train, test, cfg) {
  thr_mean   <- cfg$classification$mean_threshold    # 57.725
  thr_median <- cfg$classification$median_threshold  # 44.365

  label_split <- function(df, thr, below, above) {
    dplyr::mutate(df,
      !!rlang::sym(paste0("class_", below, "_", above)) :=
        dplyr::if_else(Price_10K < thr, below, above) |> factor()
    )
  }

  train <- train |>
    dplyr::mutate(
      class_mean   = factor(dplyr::if_else(Price_10K < thr_mean,
                                           cfg$classification$labels_mean$below,
                                           cfg$classification$labels_mean$above)),
      class_median = factor(dplyr::if_else(Price_10K < thr_median,
                                           cfg$classification$labels_median$below,
                                           cfg$classification$labels_median$above))
    )

  test <- test |>
    dplyr::mutate(
      class_mean   = factor(dplyr::if_else(Price_10K < thr_mean,
                                           cfg$classification$labels_mean$below,
                                           cfg$classification$labels_mean$above),
                            levels = levels(train$class_mean)),
      class_median = factor(dplyr::if_else(Price_10K < thr_median,
                                           cfg$classification$labels_median$below,
                                           cfg$classification$labels_median$above),
                            levels = levels(train$class_median))
    )

  message(sprintf("[add_classification_labels] Mean split (%.3f): %s",
                  thr_mean, paste(table(train$class_mean), collapse = " / ")))
  message(sprintf("[add_classification_labels] Median split (%.3f): %s",
                  thr_median, paste(table(train$class_median), collapse = " / ")))

  list(train = train, test = test)
}
