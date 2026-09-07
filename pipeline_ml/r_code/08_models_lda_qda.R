# =============================================================================
# 08_models_lda_qda.R
# Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA).
# Replicates Section: LDA & QDA from the paper.
# Both binary class targets are supported (class_mean / class_median).
# Priors: equal (0.5, 0.5) found best in paper; flat priors also tested.
# =============================================================================

suppressPackageStartupMessages({
  library(MASS)     # lda(), qda()
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------------
# Helper: select numeric predictors only (LDA/QDA require numeric input)
# ---------------------------------------------------------------------------
.disc_predictors <- function(df, class_col = "class_mean") {
  exclude <- c("Price_10K", "class_mean", "class_median")
  df %>%
    dplyr::select(-dplyr::all_of(exclude)) %>%
    dplyr::select(where(is.numeric))
}

# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------

#' Fit LDA model
#'
#' @param train     Training tibble with class labels.
#' @param class_col Response column ("class_mean" or "class_median").
#' @param priors    Numeric vector of class priors, or NULL for flat.
#' @return Fitted MASS::lda object.
fit_lda <- function(train, class_col = "class_mean", priors = c(0.5, 0.5)) {
  X <- .disc_predictors(train, class_col)
  y <- train[[class_col]]

  prior_arg <- if (!is.null(priors)) priors else NULL
  model <- MASS::lda(X, grouping = y, prior = prior_arg)

  message(sprintf("[fit_lda] LDA fitted | class_col: %s | priors: %s",
                  class_col,
                  if (is.null(prior_arg)) "flat" else paste(prior_arg, collapse = "/") ))
  model
}

#' Evaluate LDA on test set
#'
#' @param model     Fitted lda object.
#' @param test      Testing tibble.
#' @param class_col Response column.
#' @return List: $pred, $correct_rate, $confusion.
eval_lda <- function(model, test, class_col = "class_mean") {
  X_test  <- .disc_predictors(test, class_col)
  y_test  <- test[[class_col]]
  pred    <- predict(model, X_test)$class
  correct <- mean(pred == y_test)
  conf    <- table(Predicted = pred, Actual = y_test)

  message(sprintf("[eval_lda] Correct classification rate = %.7f", correct))
  list(pred = pred, correct_rate = correct, confusion = conf)
}

# ---------------------------------------------------------------------------
# QDA
# ---------------------------------------------------------------------------

#' Fit QDA model
#'
#' @param train     Training tibble with class labels.
#' @param class_col Response column.
#' @param priors    Numeric vector of class priors, or NULL.
#' @return Fitted MASS::qda object.
fit_qda <- function(train, class_col = "class_mean", priors = c(0.5, 0.5)) {
  X <- .disc_predictors(train, class_col)
  y <- train[[class_col]]

  prior_arg <- if (!is.null(priors)) priors else NULL
  model <- MASS::qda(X, grouping = y, prior = prior_arg)

  message(sprintf("[fit_qda] QDA fitted | class_col: %s | priors: %s",
                  class_col,
                  if (is.null(prior_arg)) "flat" else paste(prior_arg, collapse = "/")))
  model
}

#' Evaluate QDA on test set
#'
#' @param model     Fitted qda object.
#' @param test      Testing tibble.
#' @param class_col Response column.
#' @return List: $pred, $correct_rate, $confusion.
eval_qda <- function(model, test, class_col = "class_mean") {
  X_test  <- .disc_predictors(test, class_col)
  y_test  <- test[[class_col]]
  pred    <- predict(model, X_test)$class
  correct <- mean(pred == y_test)
  conf    <- table(Predicted = pred, Actual = y_test)

  message(sprintf("[eval_qda] Correct classification rate = %.7f", correct))
  list(pred = pred, correct_rate = correct, confusion = conf)
}

# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

#' Run both LDA and QDA with and without equal priors; print comparison table
#'
#' @param train     Training tibble.
#' @param test      Testing tibble.
#' @param class_col Response column.
#' @param cfg       Config list (for equal priors value).
#' @return Tibble summarising correct rates across four configurations.
compare_lda_qda <- function(train, test, class_col = "class_mean", cfg) {
  eq_priors <- cfg$models$lda$priors_equal

  configs <- list(
    list(method = "LDA", priors = NULL,      label = "LDA (flat priors)"),
    list(method = "LDA", priors = eq_priors, label = "LDA (equal priors)"),
    list(method = "QDA", priors = NULL,      label = "QDA (flat priors)"),
    list(method = "QDA", priors = eq_priors, label = "QDA (equal priors)")
  )

  results <- purrr::map_dfr(configs, function(cfg_i) {
    if (cfg_i$method == "LDA") {
      m <- fit_lda(train, class_col, priors = cfg_i$priors)
      e <- eval_lda(m, test, class_col)
    } else {
      m <- fit_qda(train, class_col, priors = cfg_i$priors)
      e <- eval_qda(m, test, class_col)
    }
    tibble::tibble(Model = cfg_i$label,
                   Correct_Rate = e$correct_rate,
                   Error_Rate   = 1 - e$correct_rate)
  })

  message("\n[compare_lda_qda] Results:")
  print(results)
  invisible(results)
}
