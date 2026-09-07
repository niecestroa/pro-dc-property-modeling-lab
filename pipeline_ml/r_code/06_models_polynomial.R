# =============================================================================
# 06_models_polynomial.R
# Polynomial regression model — matches the paper's final polynomial model.
# Powers sourced from cfg$polynomial (Bathrooms^8, Bedrooms^8, Fireplaces^4,
# AYB_age^5, EYB_age^4, Remodel_age^4) + Bathrooms×Condition interactions.
# Includes: outlier detection, VIF, and diagnostic plot generation.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(car)      # outlierTest, vif
})

#' Build the polynomial regression formula dynamically from config
#'
#' @param cfg  Config list.
#' @return A formula object for lm().
build_poly_formula <- function(cfg) {
  p <- cfg$polynomial

  # Categorical + linear terms (no polynomial needed)
  base_terms <- c(
    "Rooms",
    "Qualified",
    "Grade",
    "Kitchens",
    "Ward",
    "Latitude",
    "Longitude"
  )

  # Polynomial terms for continuous variables
  poly_terms <- c(
    sprintf("poly(Bathrooms,  %d, raw=TRUE)", p$Bathrooms_deg),
    sprintf("poly(Bedrooms,   %d, raw=TRUE)", p$Bedrooms_deg),
    sprintf("poly(Fireplaces, %d, raw=TRUE)", p$Fireplaces_deg),
    sprintf("poly(AYB_age,    %d, raw=TRUE)", p$AYB_age_deg),
    sprintf("poly(EYB_age,    %d, raw=TRUE)", p$EYB_age_deg),
    sprintf("poly(Remodel_age,%d, raw=TRUE)", p$Remodel_age_deg)
  )

  # Interaction: Bathrooms × Condition
  interaction_terms <- "Bathrooms:Condition"

  # Condition main effect
  condition_term <- "Condition"

  all_terms <- c(poly_terms, base_terms, condition_term, interaction_terms)
  formula_str <- paste("Price_10K ~", paste(all_terms, collapse = " + "))
  as.formula(formula_str)
}

#' Fit the polynomial regression model
#'
#' @param train   Training tibble.
#' @param cfg     Config list.
#' @param formula Optional formula override.
#' @return Fitted lm object.
fit_polynomial <- function(train, cfg, formula = NULL) {
  if (is.null(formula)) formula <- build_poly_formula(cfg)

  message("[fit_polynomial] Fitting polynomial regression ...")
  model <- lm(formula, data = train)

  summ <- summary(model)
  message(sprintf("[fit_polynomial] Adj. R2 = %.4f | RSE = %.4f | df = %d",
                  summ$adj.r.squared,
                  summ$sigma,
                  summ$df[2]))
  model
}

#' Detect outliers using car::outlierTest on a fitted polynomial model
#'
#' Mirrors the paper's finding of ~20 Bonferroni-significant outliers.
#'
#' @param model  Fitted lm from fit_polynomial().
#' @param cutoff Bonferroni p-value threshold. Default 0.05.
#' @return Integer vector of outlier row indices.
detect_outliers <- function(model, cutoff = 0.05) {
  message("[detect_outliers] Running Bonferroni outlier test ...")
  ot <- car::outlierTest(model, cutoff = cutoff, n.max = 100)
  outlier_idx <- as.integer(names(ot$bonf.p))
  message(sprintf("[detect_outliers] Found %d Bonferroni outliers (p < %.2f).",
                  length(outlier_idx), cutoff))
  outlier_idx
}

#' Remove outliers and refit the polynomial model
#'
#' @param train      Training tibble.
#' @param cfg        Config list.
#' @param outlier_idx Integer vector of row indices to remove.
#' @return List: $model (refitted lm), $train_clean (data without outliers).
refit_without_outliers <- function(train, cfg, outlier_idx) {
  n_before <- nrow(train)
  train_clean <- train[-outlier_idx, ]
  message(sprintf("[refit_without_outliers] Removed %d rows. Refitting on %d observations.",
                  n_before - nrow(train_clean), nrow(train_clean)))

  model <- fit_polynomial(train_clean, cfg)
  list(model = model, train_clean = train_clean)
}

#' Generate the four standard diagnostic plots for a polynomial model
#'
#' @param model   Fitted lm.
#' @param cfg     Config list.
#' @param tag     File name tag (e.g. "polynomial_diagnostics").
save_diagnostic_plots <- function(model, cfg, tag = "polynomial_diagnostics") {
  plot_dir <- here::here(cfg$paths$plot_dir)
  dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)
  out_file <- file.path(plot_dir, paste0(tag, ".png"))

  png(out_file, width = 1200, height = 1000, res = 120)
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  plot(model, which = 1:4, ask = FALSE)
  dev.off()

  message(sprintf("[save_diagnostic_plots] Saved → %s", out_file))
  invisible(out_file)
}
