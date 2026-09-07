# =============================================================================
# 12_models_random_forest.R
# Random Forest & Bagging for DC Property ML (Regression)
# =============================================================================

suppressPackageStartupMessages({
  library(ranger)
  library(randomForest)
  library(dplyr)
  library(ggplot2)
})

# ---------------------------------------------------------------------------
# Helper: numeric predictor matrix
# ---------------------------------------------------------------------------

.rf_Xy <- function(df, response = "Price_10K") {
  X <- dplyr::select(df, -dplyr::all_of(response))
  y <- df[[response]]
  list(X = X, y = y)
}

# ---------------------------------------------------------------------------
# [A] Random Forest (RANGER) — Default
# ---------------------------------------------------------------------------

fit_random_forest <- function(train, cfg) {
  ntree_values <- cfg$models$random_forest$ntree_values
  mtry_range   <- cfg$models$random_forest$mtry_range
  
  xy <- .rf_Xy(train)
  
  message("[fit_random_forest] Fitting RANGER RF models...")
  
  models <- list()
  results <- list()
  
  for (nt in ntree_values) {
    for (m in mtry_range) {
      message(sprintf("  - ntree=%d | mtry=%d", nt, m))
      
      rf <- ranger(
        formula = Price_10K ~ .,
        data    = train,
        num.trees = nt,
        mtry      = m,
        importance = ifelse(cfg$models$random_forest$importance, "impurity", "none")
      )
      
      pred <- predict(rf, data = train)$predictions
      mse  <- mean((train$Price_10K - pred)^2)
      
      models[[paste(nt, m, sep = "_")]] <- rf
      results[[paste(nt, m, sep = "_")]] <- mse
    }
  }
  
  best_key <- names(results)[which.min(unlist(results))]
  best_model <- models[[best_key]]
  
  message(sprintf("[fit_random_forest] Best model: %s | MSE=%.4f",
                  best_key, min(unlist(results))))
  
  best_model
}

# ---------------------------------------------------------------------------
# Evaluate RF (default)
# ---------------------------------------------------------------------------

eval_random_forest <- function(model, test) {
  pred <- predict(model, data = test)$predictions
  mse  <- mean((test$Price_10K - pred)^2)
  
  message(sprintf("[eval_random_forest] Test MSE = %.6f", mse))
  list(pred = pred, mse = mse)
}

# ---------------------------------------------------------------------------
# [B] Bagging (randomForest)
# ---------------------------------------------------------------------------

fit_bagging <- function(train, cfg) {
  nt <- cfg$models$bagging$ntree
  
  message(sprintf("[fit_bagging] Fitting Bagging model (ranger) | ntree=%d", nt))
  
  model <- ranger(
    formula = Price_10K ~ .,
    data = train,
    num.trees = nt,
    mtry = ncol(train) - 1,   # full bagging
    importance = "impurity"
  )
  
  message("[fit_bagging] Model fitted.")
  model
}

# ---------------------------------------------------------------------------
# [C] Optimal RF (best ntree + best mtry)
# ---------------------------------------------------------------------------

fit_rf_optimal <- function(train, cfg) {
  ntree_values <- cfg$models$random_forest$ntree_values
  mtry_range   <- cfg$models$random_forest$mtry_range
  
  xy <- .rf_Xy(train)
  
  message("[fit_rf_optimal] Searching optimal RF...")
  
  best_mse <- Inf
  best_model <- NULL
  best_nt <- NA
  best_m <- NA
  
  for (nt in ntree_values) {
    for (m in mtry_range) {
      rf <- ranger(
        formula = Price_10K ~ .,
        data    = train,
        num.trees = nt,
        mtry      = m,
        importance = ifelse(cfg$models$random_forest$importance, "impurity", "none")
      )
      
      pred <- predict(rf, data = train)$predictions
      mse  <- mean((train$Price_10K - pred)^2)
      
      if (mse < best_mse) {
        best_mse <- mse
        best_model <- rf
        best_nt <- nt
        best_m <- m
      }
    }
  }
  
  message(sprintf("[fit_rf_optimal] Best ntree=%d | mtry=%d | MSE=%.6f",
                  best_nt, best_m, best_mse))
  
  best_model
}

eval_rf_optimal <- function(model, test) {
  pred <- predict(model, data = test)$predictions
  mse  <- mean((test$Price_10K - pred)^2)
  
  message(sprintf("[eval_rf_optimal] Test MSE = %.6f", mse))
  list(pred = pred, mse = mse)
}

# ---------------------------------------------------------------------------
# [D] Plots
# ---------------------------------------------------------------------------

save_rf_error_plot <- function(model, cfg, tag = "rf_oob_error") {
  oob_pred   <- model$predictions
  oob_actual <- model$training.data$Price_10K
  oob_mse    <- (oob_actual - oob_pred)^2
  
  df <- data.frame(
    Trees = seq_along(oob_mse),
    OOB_MSE = oob_mse
  )
  
  p <- ggplot(df, aes(x = Trees, y = OOB_MSE)) +
    geom_line(color = "steelblue", linewidth = 1) +
    labs(
      title = paste("Random Forest OOB Error —", tag),
      x = "Trees",
      y = "OOB MSE"
    ) +
    theme_minimal(base_size = 14)
  
  out_path <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  ggsave(out_path, p, width = 8, height = 6)
  message("[save_rf_error_plot] Saved → ", out_path)
}

save_importance_plot <- function(model, cfg, tag = "rf_importance") {
  if ("ranger" %in% class(model)) {
    imp <- data.frame(
      Variable   = names(model$variable.importance),
      Importance = model$variable.importance
    )
    
    p <- ggplot(imp, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(
        title = paste("Variable Importance —", tag),
        x = "Predictor",
        y = "Importance"
      ) +
      theme_minimal(base_size = 14)
    
  } else if ("randomForest" %in% class(model)) {
    imp <- data.frame(
      Variable   = rownames(model$importance),
      Importance = model$importance[, 1]
    )
    
    p <- ggplot(imp, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_col(fill = "darkgreen") +
      coord_flip() +
      labs(
        title = paste("Variable Importance —", tag),
        x = "Predictor",
        y = "IncNodePurity"
      ) +
      theme_minimal(base_size = 14)
    
  } else {
    stop("Unsupported RF model class.")
  }
  
  out_path <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  ggsave(out_path, p, width = 8, height = 6)
  message("[save_importance_plot] Saved → ", out_path)
}

# ---------------------------------------------------------------------------
# Modern RF Evaluation (RMSE, MAE, R2) — FIXED VERSION
# ---------------------------------------------------------------------------

eval_rf_modern <- function(model, test, response = "Price_10K") {
  
  # Force ranger to use correct prediction path
  preds <- predict(model, data = test)$predictions
  
  # Safety check: prevent silent numeric(0) failures
  if (length(preds) == 0) {
    stop("[eval_rf_modern] ERROR: Ranger returned zero predictions. Train/test column mismatch.")
  }
  
  actual <- test[[response]]
  
  mse  <- mean((actual - preds)^2)
  rmse <- sqrt(mse)
  mae  <- mean(abs(actual - preds))
  r2   <- 1 - sum((actual - preds)^2) / sum((actual - mean(actual))^2)
  
  list(
    pred = preds,
    MSE  = mse,
    RMSE = rmse,
    MAE  = mae,
    R2   = r2
  )
}

# ---------------------------------------------------------------------------
# Alias for pipeline compatibility
# ---------------------------------------------------------------------------

eval_rf <- eval_rf_modern
