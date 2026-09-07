# =============================================================================
# regression_metrics.R
# Compute standard regression metrics for model evaluation.
# =============================================================================

regression_metrics <- function(actual, predicted, model_name = "Model") {
  
  # Handle NA predictions (common with aliased lm coefficients)
  keep <- !is.na(predicted)
  actual <- actual[keep]
  predicted <- predicted[keep]
  
  mse  <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  mae  <- mean(abs(actual - predicted))
  r2   <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  
  tibble::tibble(
    Model = model_name,
    MSE   = mse,
    RMSE  = rmse,
    MAE   = mae,
    R2    = r2
  )
}
