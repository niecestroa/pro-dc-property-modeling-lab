# =============================================================================
# 01_load_config.R
# Load and validate the project YAML configuration.
# Returns a named list used by every downstream script.
# =============================================================================

suppressPackageStartupMessages(library(yaml))

#' Load project configuration from config/config.yaml
#'
#' @param config_path Path to the YAML config file.
#' @return A named list of configuration values.
load_config <- function(config_path = here::here("config", "config.yaml")) {
  if (!file.exists(config_path)) {
    stop(sprintf("[load_config] Config file not found: %s", config_path))
  }
  cfg <- yaml::read_yaml(config_path)
  message(sprintf("[load_config] Loaded config v%s for project: '%s'",
                  cfg$project$version, cfg$project$name))
  cfg
}

#' Set the global random seed from config
#'
#' @param cfg Config list returned by load_config().
set_project_seed <- function(cfg) {
  seed <- cfg$project$seed
  set.seed(seed)
  message(sprintf("[set_project_seed] Random seed set to %d", seed))
}

#' Helper: resolve a path key from config (relative to project root)
#'
#' @param cfg Config list.
#' @param key  Character key inside cfg$paths (e.g. "raw_data").
#' @return Absolute path string.
cfg_path <- function(cfg, key) {
  here::here(cfg$paths[[key]])
}
