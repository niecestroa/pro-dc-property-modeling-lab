# =============================================================================
# 10_models_pcr_pls.R — PCR & PLS (fully compatible with older dplyr)
# =============================================================================

suppressPackageStartupMessages({
  library(pls)
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------------
# Helper: safe numeric predictor selection (no tidyselect)
# ---------------------------------------------------------------------------

get_numeric_predictors <- function(train) {
  num_cols <- names(train)[sapply(train, is.numeric)]
  num_cols <- setdiff(num_cols, c("Price_10K"))
  num_cols <- num_cols[!grepl("^class_", num_cols)]
  num_cols
}

# ---------------------------------------------------------------------------
# PCR
# ---------------------------------------------------------------------------

fit_pcr <- function(train, cfg, formula = NULL) {
  
  ncomp_max  <- cfg$models$pcr$ncomp_max
  validation <- cfg$models$pcr$validation
  
  if (is.null(formula)) {
    num_preds <- get_numeric_predictors(train)
    formula <- as.formula(paste("Price_10K ~", paste(num_preds, collapse = " + ")))
  }
  
  message(sprintf("[fit_pcr] Fitting PCR (ncomp=%d, validation=%s) ...",
                  ncomp_max, validation))
  
  model <- pls::pcr(formula,
                    data       = train,
                    ncomp      = ncomp_max,
                    validation = validation,
                    scale      = TRUE)
  
  cv_rmsep   <- pls::RMSEP(model, estimate = "CV")
  best_ncomp <- which.min(cv_rmsep$val[1, 1, ]) - 1L
  
  if (is.na(best_ncomp) || best_ncomp < 1L) best_ncomp <- 1L
  
  message(sprintf("[fit_pcr] Best ncomp (min CV RMSEP) = %d", best_ncomp))
  attr(model, "best_ncomp") <- best_ncomp
  model
}

eval_pcr <- function(model, test, ncomp = NULL) {
  
  if (is.null(ncomp)) ncomp <- attr(model, "best_ncomp")
  if (is.na(ncomp) || ncomp < 1L) ncomp <- 1L
  
  pred <- as.numeric(predict(model, newdata = test, ncomp = ncomp))
  mse  <- mean((pred - test$Price_10K)^2)
  
  message(sprintf("[eval_pcr] Test MSE (ncomp=%d) = %.4f", ncomp, mse))
  list(pred = pred, mse = mse, ncomp_used = ncomp)
}

# ---------------------------------------------------------------------------
# PLS
# ---------------------------------------------------------------------------

fit_pls <- function(train, cfg, formula = NULL) {
  
  ncomp_max  <- cfg$models$pls$ncomp_max
  validation <- cfg$models$pls$validation
  
  if (is.null(formula)) {
    num_preds <- get_numeric_predictors(train)
    formula <- as.formula(paste("Price_10K ~", paste(num_preds, collapse = " + ")))
  }
  
  message(sprintf("[fit_pls] Fitting PLS (ncomp=%d, validation=%s) ...",
                  ncomp_max, validation))
  
  model <- pls::plsr(formula,
                     data       = train,
                     ncomp      = ncomp_max,
                     validation = validation,
                     scale      = TRUE)
  
  cv_rmsep   <- pls::RMSEP(model, estimate = "CV")
  best_ncomp <- which.min(cv_rmsep$val[1, 1, ]) - 1L
  
  if (is.na(best_ncomp) || best_ncomp < 1L) best_ncomp <- 1L
  
  message(sprintf("[fit_pls] Best ncomp (min CV RMSEP) = %d", best_ncomp))
  attr(model, "best_ncomp") <- best_ncomp
  model
}

eval_pls <- function(model, test, ncomp = NULL) {
  
  if (is.null(ncomp)) ncomp <- attr(model, "best_ncomp")
  if (is.na(ncomp) || ncomp < 1L) ncomp <- 1L
  
  pred <- as.numeric(predict(model, newdata = test, ncomp = ncomp))
  mse  <- mean((pred - test$Price_10K)^2)
  
  message(sprintf("[eval_pls] Test MSE (ncomp=%d) = %.4f", ncomp, mse))
  list(pred = pred, mse = mse, ncomp_used = ncomp)
}

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

save_rmsep_plot <- function(model, cfg, tag = "pcr_rmsep") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  png(out, width = 900, height = 600, res = 120)
  pls::validationplot(model, val.type = "RMSEP",
                      main = paste(toupper(gsub("_.*", "", tag)), "— RMSEP vs. Components"))
  dev.off()
  message(sprintf("[save_rmsep_plot] Saved → %s", out))
}

save_variance_plot <- function(model, cfg, tag = "pcr_variance") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  png(out, width = 900, height = 600, res = 120)
  pls::explvar(model) |>
    cumsum() |>
    plot(type = "b", xlab = "Components", ylab = "Cumulative Variance Explained (%)",
         main = paste(toupper(gsub("_.*", "", tag)), "— Variance Explained"))
  dev.off()
  message(sprintf("[save_variance_plot] Saved → %s", out))
}
