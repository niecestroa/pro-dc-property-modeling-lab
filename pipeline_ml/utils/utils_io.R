# =============================================================================
# utils/utils_io.R
# I/O helpers: model serialisation, table export, directory management.
# =============================================================================

suppressPackageStartupMessages(library(readr))

#' Save a fitted model object to the models/ directory
#'
#' @param model    Any R model object.
#' @param cfg      Config list.
#' @param name     Filename base (no extension), e.g. "ridge", "rf_optimal".
save_model <- function(model, cfg, name) {
  dir <- here::here(cfg$paths$model_dir)
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(dir, paste0(name, ".rds"))
  saveRDS(model, path)
  message(sprintf("[save_model] Saved '%s' → %s", name, path))
  invisible(path)
}

#' Load a previously saved model object
#'
#' @param cfg  Config list.
#' @param name Model name (same as used in save_model()).
#' @return Deserialized R model object.
load_model <- function(cfg, name) {
  path <- file.path(here::here(cfg$paths$model_dir), paste0(name, ".rds"))
  if (!file.exists(path)) stop(sprintf("[load_model] Not found: %s", path))
  message(sprintf("[load_model] Loading '%s' from %s", name, path))
  readRDS(path)
}

#' Ensure all output directories defined in cfg$paths exist
#'
#' @param cfg Config list.
ensure_output_dirs <- function(cfg) {
  dirs <- c(
    here::here(cfg$paths$model_dir),
    here::here(cfg$paths$plot_dir),
    here::here(cfg$paths$table_dir),
    here::here(cfg$paths$report_dir),
    here::here(dirname(cfg$paths$processed_data)),
    here::here(dirname(cfg$paths$train_data))
  )
  for (d in unique(dirs)) dir.create(d, showWarnings = FALSE, recursive = TRUE)
  message(sprintf("[ensure_output_dirs] %d output directories verified.", length(unique(dirs))))
}

#' Append a row to the run log CSV
#'
#' @param cfg       Config list.
#' @param model     Model name string.
#' @param metric    Metric name, e.g. "Test_MSE" or "Correct_Rate".
#' @param value     Numeric value.
#' @param notes     Optional notes string.
log_result <- function(cfg, model, metric, value, notes = "") {
  log_path <- file.path(here::here(cfg$paths$table_dir), "run_log.csv")
  row <- data.frame(
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    model     = model,
    metric    = metric,
    value     = round(value, 8),
    notes     = notes,
    stringsAsFactors = FALSE
  )
  if (file.exists(log_path)) {
    readr::write_csv(row, log_path, append = TRUE)
  } else {
    readr::write_csv(row, log_path)
  }
}
