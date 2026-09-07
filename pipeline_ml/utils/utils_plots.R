# =============================================================================
# utils/utils_plots.R
# Reusable plotting helpers for EDA and model diagnostics.
# All plots saved to cfg$paths$plot_dir automatically.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

#' Histogram of Price_10K with mean and median lines
save_price_histogram <- function(df, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir), "eda_price_histogram.png")
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  
  mn <- mean(df$Price_10K, na.rm = TRUE)
  md <- median(df$Price_10K, na.rm = TRUE)
  
  p <- ggplot(df, aes(x = Price_10K)) +
    geom_histogram(bins = 80, fill = "steelblue", colour = "white", alpha = 0.85) +
    geom_vline(xintercept = mn, colour = "firebrick", linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = md, colour = "darkgreen", linetype = "dashed", linewidth = 1) +
    annotate("text", x = mn + 5, y = Inf, vjust = 2,
             label = sprintf("Mean=%.3f", mn), colour = "firebrick", size = 3.5) +
    annotate("text", x = md - 5, y = Inf, vjust = 4,
             label = sprintf("Median=%.3f", md), colour = "darkgreen", size = 3.5,
             hjust = 1) +
    labs(title = "DC Property — Price Distribution (in $10K)",
         x = "Price ($10K)", y = "Count") +
    theme_bw()
  
  ggsave(out, p, width = 9, height = 5, dpi = 150)
  message(sprintf("[save_price_histogram] Saved => %s", out))
  invisible(p)
}

#' Boxplots of Price_10K by a categorical variable
save_price_by_category <- function(df, cat_var, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir),
                   paste0("eda_price_by_", tolower(cat_var), ".png"))
  
  p <- ggplot(df, aes(x = .data[[cat_var]], y = Price_10K, fill = .data[[cat_var]])) +
    geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3, show.legend = FALSE) +
    labs(title = paste("Price_10K by", cat_var),
         x = cat_var, y = "Price ($10K)") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  
  ggsave(out, p, width = 9, height = 5, dpi = 150)
  message(sprintf("[save_price_by_category] Saved => %s", out))
  invisible(p)
}

#' Scatter matrix of key numeric predictors vs Price_10K
save_scatter_matrix <- function(df, cfg) {
  if (!requireNamespace("GGally", quietly = TRUE)) {
    warning("[save_scatter_matrix] Install GGally: install.packages('GGally')")
    return(invisible(NULL))
  }
  out <- file.path(here::here(cfg$paths$plot_dir), "eda_scatter_matrix.png")
  vars <- c("Price_10K", "Bathrooms", "Rooms", "Bedrooms",
            "Fireplaces", "AYB_age", "EYB_age")
  vars <- intersect(vars, colnames(df))
  
  p <- GGally::ggpairs(df[, vars], progress = FALSE,
                       upper = list(continuous = GGally::wrap("cor", size = 3)),
                       lower = list(continuous = GGally::wrap("points", alpha = 0.1, size = 0.3)),
                       title = "Scatter Matrix — Key Predictors vs Price_10K")
  
  ggsave(out, p, width = 12, height = 12, dpi = 120)
  message(sprintf("[save_scatter_matrix] Saved => %s", out))
  invisible(p)
}

#' Geographic scatter: Lat × Lon coloured by Price_10K
save_geo_price_map <- function(df, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir), "eda_geo_price_map.png")
  
  p <- ggplot(
    df %>% sample_n(min(10000, nrow(df))),
    aes(x = Longitude, y = Latitude, colour = Price_10K)
  ) +
    geom_point(alpha = 0.5, size = 0.6) +
    scale_colour_viridis_c(option = "plasma", name = "Price ($10K)") +
    labs(title = "DC Property Prices — Geographic Distribution") +
    theme_bw()
  
  ggsave(out, p, width = 8, height = 7, dpi = 150)
  message(sprintf("[save_geo_price_map] Saved => %s", out))
  invisible(p)
}

#' Correlation heatmap for numeric predictors
save_correlation_heatmap <- function(df, cfg) {
  out <- file.path(here::here(cfg$paths$plot_dir), "eda_correlation_heatmap.png")
  
  num_df <- df %>%
    dplyr::select_if(is.numeric) %>%
    dplyr::select(-dplyr::matches("^class_"))
  
  cor_mat <- cor(num_df, use = "pairwise.complete.obs")
  cor_long <- as.data.frame(as.table(cor_mat)) %>%
    rename(Var1 = Var1, Var2 = Var2, Correlation = Freq)
  
  p <- ggplot(cor_long, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile(colour = "white") +
    scale_fill_gradient2(low = "firebrick", high = "steelblue",
                         mid = "white", midpoint = 0, limits = c(-1, 1)) +
    labs(title = "Correlation Heatmap — Numeric Predictors") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title  = element_blank())
  
  ggsave(out, p, width = 10, height = 9, dpi = 150)
  message(sprintf("[save_correlation_heatmap] Saved => %s", out))
  invisible(p)
}

# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

#' Save residual plots for an lm model
save_residual_plots <- function(model, cfg, tag = "linear_diagnostics") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  png(out, width = 1200, height = 600, res = 120)
  par(mfrow = c(1, 2))
  plot(model, which = c(1, 2), ask = FALSE)
  dev.off()
  message(sprintf("[save_residual_plots] Saved => %s", out))
}

#' Save Predicted vs Actual Plot
save_pred_vs_actual <- function(actual, pred, cfg, tag = "pred_vs_actual") {
  df <- data.frame(
    Actual = actual,
    Predicted = pred
  )
  
  p <- ggplot(df, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6, color = "#2C3E50") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(
      title = paste("Predicted vs Actual —", tag),
      x = "Actual Price (10K USD)",
      y = "Predicted Price (10K USD)"
    ) +
    theme_minimal(base_size = 14)
  
  out_path <- file.path(here::here(cfg$paths$plot_dir),
                        paste0("pred_vs_actual_", tag, ".png"))
  
  ggsave(out_path, p, width = 8, height = 6)
  message("[save_pred_vs_actual] Saved =>", out_path)
}

save_classification_bar_chart <- function(cls_table, cfg, tag = "classification_bar") {
  out <- file.path(here::here(cfg$paths$plot_dir), paste0(tag, ".png"))
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  
  p <- ggplot(cls_table, aes(x = Model, y = Accuracy, fill = Model)) +
    geom_col(alpha = 0.8) +
    geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5, size = 4) +
    labs(
      title = paste("Classification Accuracy —", tag),
      x = "Model",
      y = "Accuracy"
    ) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none")
  
  ggsave(out, p, width = 8, height = 6, dpi = 150)
  message("[save_classification_bar_chart] Saved =>", out)
}

classification_comparison_table <- function(results_list) {
  models <- names(results_list)
  
  df <- lapply(models, function(m) {
    actual <- results_list[[m]]$actual
    pred   <- results_list[[m]]$pred
    
    accuracy <- mean(actual == pred)
    misclass <- 1 - accuracy
    
    data.frame(
      Model = m,
      Accuracy = round(accuracy, 4),
      Misclassification_Rate = round(misclass, 4)
    )
  })
  
  do.call(rbind, df)
}

save_comparison_table <- function(tbl, cfg, tag) {
  out <- file.path(
    here::here(cfg$paths$plot_dir),
    paste0(tag, "_table.csv")
  )
  
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  write.csv(tbl, out, row.names = FALSE)
  
  message("[save_comparison_table] Saved =>", out)
}

save_confusion_matrix <- function(actual, pred, cfg, tag = "confusion_matrix") {
  # Build confusion matrix
  cm <- table(Actual = actual, Predicted = pred)
  cm_df <- as.data.frame(cm)
  
  # Plot
  p <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 5) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(
      title = paste("Confusion Matrix —", tag),
      x = "Actual",
      y = "Predicted"
    ) +
    theme_minimal(base_size = 14)
  
  # Save
  out <- file.path(
    here::here(cfg$paths$plot_dir),
    paste0(tag, ".png")
  )
  
  ggsave(out, p, width = 6, height = 5, dpi = 150)
  message("[save_confusion_matrix] Saved => ", out)
}

