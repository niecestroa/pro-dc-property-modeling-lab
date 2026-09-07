# =============================================================================
# 11_models_trees.R
# Regression Trees with cost-complexity pruning.
# Matches paper's Section: Regression Trees.
# Variables used in paper's tree: Longitude, Bathrooms, EYB_age,
#   Fireplaces, AYB_age, Latitude (11 terminal nodes).
# =============================================================================

suppressPackageStartupMessages({
  library(rpart)
  library(rpart.plot)
  library(dplyr)
})

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

#' Fit a regression tree (anova method)
#'
#' @param train   Training tibble.
#' @param formula Model formula. Defaults to all available predictors.
#' @param cfg     Config list.
#' @return Fitted rpart object (unpruned).
fit_tree <- function(train, cfg, formula = NULL) {
  if (is.null(formula)) {
    exclude <- c("class_mean", "class_median")
    preds   <- setdiff(colnames(train), c("Price_10K", exclude))
    formula <- as.formula(paste("Price_10K ~", paste(preds, collapse = " + ")))
  }

  message("[fit_tree] Fitting regression tree (method = anova) ...")
  model <- rpart::rpart(
    formula,
    data    = train,
    method  = cfg$models$tree$method,   # "anova"
    control = rpart::rpart.control(xval = 10, cp = 0.001)
  )

  message(sprintf("[fit_tree] Tree has %d terminal nodes.",
                  sum(model$frame$var == "<leaf>")))
  model
}

# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

#' Prune a regression tree using the 1-SE or minimum cross-validation error rule
#'
#' @param tree    Fitted rpart object from fit_tree().
#' @param cfg     Config list ($models$tree$cp_prune: "min" or "1se").
#' @return Pruned rpart object.
prune_tree <- function(tree, cfg) {
  method <- cfg$models$tree$cp_prune    # "min"
  cp_table <- tree$cptable

  if (method == "1se") {
    # 1-SE rule: smallest tree with error ≤ min(xerror) + xstd at min
    min_idx  <- which.min(cp_table[, "xerror"])
    threshold <- cp_table[min_idx, "xerror"] + cp_table[min_idx, "xstd"]
    cp_opt   <- cp_table[min(which(cp_table[, "xerror"] <= threshold)), "CP"]
  } else {
    # minimum xerror
    cp_opt <- cp_table[which.min(cp_table[, "xerror"]), "CP"]
  }

  message(sprintf("[prune_tree] Optimal CP = %.6f (method = '%s')", cp_opt, method))
  pruned <- rpart::prune(tree, cp = cp_opt)
  message(sprintf("[prune_tree] Pruned tree has %d terminal nodes.",
                  sum(pruned$frame$var == "<leaf>")))
  pruned
}

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

#' Compute test MSE for a regression tree
#'
#' @param model   rpart object (pruned or unpruned).
#' @param test    Testing tibble.
#' @return List: $pred, $mse.
eval_tree <- function(model, test) {
  pred <- predict(model, newdata = test)
  mse  <- mean((pred - test$Price_10K)^2)
  message(sprintf("[eval_tree] Test MSE = %.4f", mse))
  list(pred = pred, mse = mse)
}

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

#' Save the regression tree diagram
#'
#' @param model rpart object.
#' @param cfg   Config list.
#' @param tag   File tag, e.g. "tree_unpruned" or "tree_pruned".
save_tree_plot <- function(model, cfg, tag = "tree_pruned") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  png(out, width = 1400, height = 900, res = 120)
  rpart.plot::rpart.plot(
    model,
    type    = 4,
    extra   = 101,
    under   = TRUE,
    fallen.leaves = TRUE,
    main    = paste("Regression Tree —", gsub("_", " ", tag)),
    cex     = 0.7
  )
  dev.off()
  message(sprintf("[save_tree_plot] Saved → %s", out))
}

#' Save the CP (complexity parameter) table plot for pruning visualisation
#'
#' @param model rpart object.
#' @param cfg   Config list.
save_cp_plot <- function(model, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir), "tree_cp_plot.png")
  png(out, width = 900, height = 600, res = 120)
  rpart::plotcp(model, main = "Regression Tree — CP vs. Cross-Validation Error")
  dev.off()
  message(sprintf("[save_cp_plot] Saved → %s", out))
}
