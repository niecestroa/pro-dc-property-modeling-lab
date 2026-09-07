# =============================================================================
# 04_feature_engineering.R
# Build polynomial terms and interaction features that match the final
# polynomial model from the original paper (Section: Polynomial Regression).
#
# All powers come from config$polynomial so they can be tuned without
# editing code. A "raw" design matrix helper is also provided for glmnet.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(stats)
})

# -----------------------------------------------------------------------------
# Internal helper: add poly columns for one variable
# -----------------------------------------------------------------------------
.add_poly <- function(df, var, degree, center_scale = FALSE) {
  if (!var %in% colnames(df)) {
    warning(sprintf("[feature_engineering] Column '%s' not found — skipping.", var))
    return(df)
  }
  poly_mat <- poly(df[[var]], degree = degree, raw = TRUE)
  col_names <- paste0(var, "_p", seq_len(degree))
  poly_df   <- as.data.frame(poly_mat)
  colnames(poly_df) <- col_names
  # Degree 1 duplicates the original; skip it to avoid multicollinearity
  poly_df <- poly_df[, -1, drop = FALSE]
  dplyr::bind_cols(df, poly_df)
}

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

#' Add all polynomial and interaction features used in the original paper
#'
#' Polynomial degrees sourced from cfg$polynomial:
#'   Bathrooms  → p2–p8
#'   Bedrooms   → p2–p8
#'   Fireplaces → p2–p4
#'   AYB_age    → p2–p5
#'   EYB_age    → p2–p4
#'   Remodel_age→ p2–p4
#'
#' Interaction:  Bathrooms × Condition dummies (Excellent/Fair/Good/Very Good)
#'
#' @param df  A split tibble (train or test).
#' @param cfg Config list.
#' @return Tibble with original columns plus all polynomial/interaction columns.
add_polynomial_features <- function(df, cfg) {
  p <- cfg$polynomial

  df <- .add_poly(df, "Bathrooms",   p$Bathrooms_deg)
  df <- .add_poly(df, "Bedrooms",    p$Bedrooms_deg)
  df <- .add_poly(df, "Fireplaces",  p$Fireplaces_deg)
  df <- .add_poly(df, "AYB_age",     p$AYB_age_deg)
  df <- .add_poly(df, "EYB_age",     p$EYB_age_deg)
  df <- .add_poly(df, "Remodel_age", p$Remodel_age_deg)

  # Interaction: Bathrooms × Condition levels (paper uses 4 condition levels)
  if ("Condition" %in% colnames(df) && "Bathrooms" %in% colnames(df)) {
    cond_levels <- c("Excellent", "Fair", "Good", "Very Good")
    for (lv in cond_levels) {
      safe_name <- paste0("Bath_x_Cond_", gsub(" ", "_", lv))
      df[[safe_name]] <- df$Bathrooms * as.integer(df$Condition == lv)
    }
  }

  message(sprintf("[add_polynomial_features] Feature matrix: %d cols", ncol(df)))
  df
}

#' Build a numeric model matrix suitable for glmnet (Ridge / Lasso / PCR / PLS)
#'
#' Drops the response column, converts factors to dummy codes (no intercept).
#'
#' @param df      Tibble with raw + polynomial features.
#' @param response Name of the response column (default "Price_10K").
#' @return A numeric matrix X with no intercept column.
build_model_matrix <- function(df, response = "Price_10K") {
  formula_str <- paste(response, "~ . - 1")    # -1 → no intercept from model.matrix
  # Drop class label columns if present
  drop_cols <- grep("^class_", colnames(df), value = TRUE)
  df_x <- df[, !colnames(df) %in% c(response, drop_cols)]
  X <- model.matrix(~ . - 1, data = df_x)
  X
}

#' Build X matrix and y vector in one call
#'
#' @param df       Tibble.
#' @param response Response column name.
#' @return List $X (matrix) and $y (numeric vector).
build_Xy <- function(df, response = "Price_10K") {
  list(
    X = build_model_matrix(df, response),
    y = df[[response]]
  )
}
