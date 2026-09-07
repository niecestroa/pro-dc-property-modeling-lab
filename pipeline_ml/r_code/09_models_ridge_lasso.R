# =============================================================================
# 09_models_ridge_lasso.R
# Ridge (alpha=0) and Lasso (alpha=1) regularisation via glmnet.
# Applied to the polynomial feature matrix (matches paper's approach).
# Includes: CV lambda selection, coefficient path plots, test MSE evaluation.
# =============================================================================

suppressPackageStartupMessages({
  library(glmnet)
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------------
# Internal: run glmnet with cross-validation
# ---------------------------------------------------------------------------
.fit_glmnet_cv <- function(X_train, y_train, alpha, nfolds, label) {
  message(sprintf("[%s] Running %d-fold CV glmnet (alpha=%g) ...", label, nfolds, alpha))
  set.seed(1)   # additional seed for CV fold reproducibility
  cv_model <- glmnet::cv.glmnet(
    x       = X_train,
    y       = y_train,
    alpha   = alpha,
    nfolds  = nfolds,
    standardize = TRUE
  )
  lambda_min <- cv_model$lambda.min
  lambda_1se <- cv_model$lambda.1se
  message(sprintf("[%s] lambda.min = %.6f | lambda.1se = %.6f", label, lambda_min, lambda_1se))
  cv_model
}

# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

#' Fit Ridge regression with cross-validated lambda selection
#'
#' @param train_Xy List with $X (matrix) and $y (numeric) from build_Xy().
#' @param cfg      Config list.
#' @return cv.glmnet object (alpha=0).
fit_ridge <- function(train_Xy, cfg) {
  .fit_glmnet_cv(
    X_train = train_Xy$X,
    y_train = train_Xy$y,
    alpha   = cfg$models$ridge$alpha,    # 0
    nfolds  = cfg$models$ridge$nfolds,   # 10
    label   = "fit_ridge"
  )
}

#' Evaluate Ridge on test set using lambda.min
#'
#' @param cv_ridge  cv.glmnet object from fit_ridge().
#' @param test_Xy   List with $X and $y for the test set.
#' @param lambda    Lambda to use. Default: lambda.min.
#' @return List: $pred, $mse, $lambda_used.
eval_ridge <- function(cv_ridge, test_Xy, lambda = NULL) {
  lam  <- if (is.null(lambda)) cv_ridge$lambda.min else lambda
  pred <- as.numeric(predict(cv_ridge, newx = test_Xy$X, s = lam))
  mse  <- mean((pred - test_Xy$y)^2)
  message(sprintf("[eval_ridge] Test MSE (lambda=%.6f) = %.2f", lam, mse))
  list(pred = pred, mse = mse, lambda_used = lam)
}

# ---------------------------------------------------------------------------
# Lasso
# ---------------------------------------------------------------------------

#' Fit Lasso regression with cross-validated lambda selection
#'
#' @param train_Xy List with $X and $y from build_Xy().
#' @param cfg      Config list.
#' @return cv.glmnet object (alpha=1).
fit_lasso <- function(train_Xy, cfg) {
  .fit_glmnet_cv(
    X_train = train_Xy$X,
    y_train = train_Xy$y,
    alpha   = cfg$models$lasso$alpha,    # 1
    nfolds  = cfg$models$lasso$nfolds,
    label   = "fit_lasso"
  )
}

#' Evaluate Lasso on test set using lambda.min
#'
#' @param cv_lasso  cv.glmnet object from fit_lasso().
#' @param test_Xy   List with $X and $y for the test set.
#' @param lambda    Lambda to use. Default: lambda.min.
#' @return List: $pred, $mse, $lambda_used, $n_nonzero.
eval_lasso <- function(cv_lasso, test_Xy, lambda = NULL) {
  lam  <- if (is.null(lambda)) cv_lasso$lambda.min else lambda
  pred <- as.numeric(predict(cv_lasso, newx = test_Xy$X, s = lam))
  mse  <- mean((pred - test_Xy$y)^2)

  coefs    <- coef(cv_lasso, s = lam)
  n_nonzero <- sum(coefs[-1] != 0)   # exclude intercept

  message(sprintf("[eval_lasso] Test MSE (lambda=%.6f) = %.2f | Non-zero coefs: %d",
                  lam, mse, n_nonzero))
  list(pred = pred, mse = mse, lambda_used = lam, n_nonzero = n_nonzero)
}

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

#' Save the CV error vs. log(lambda) plot for Ridge or Lasso
#'
#' @param cv_model cv.glmnet object.
#' @param cfg      Config list.
#' @param tag      File tag, e.g. "ridge_cv" or "lasso_cv".
save_cv_plot <- function(cv_model, cfg, tag = "ridge_cv") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, "_lambda.png"))
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  png(out, width = 900, height = 600, res = 120)
  plot(cv_model, main = paste(toupper(tag), "— CV Error vs. log(lambda)"))
  dev.off()
  message(sprintf("[save_cv_plot] Saved → %s", out))
}

#' Save the coefficient path (L1-norm) plot for Ridge or Lasso
#'
#' @param cv_model cv.glmnet object.
#' @param cfg      Config list.
#' @param tag      File tag.
save_coef_path_plot <- function(cv_model, cfg, tag = "ridge_coef_path") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  png(out, width = 900, height = 600, res = 120)
  plot(cv_model$glmnet.fit, xvar = "norm",
       main = paste(toupper(tag), "— Coefficient Path (L1 Norm)"))
  dev.off()
  message(sprintf("[save_coef_path_plot] Saved → %s", out))
}

#' Compare Ridge vs Lasso test MSE in a tidy tibble
#'
#' @param ridge_eval Output of eval_ridge().
#' @param lasso_eval Output of eval_lasso().
#' @return Tibble with columns Model and Test_MSE.
compare_ridge_lasso <- function(ridge_eval, lasso_eval) {
  result <- tibble::tibble(
    Model    = c("Ridge", "Lasso"),
    Test_MSE = c(ridge_eval$mse, lasso_eval$mse)
  )
  message("\n[compare_ridge_lasso]")
  print(result)
  invisible(result)
}
