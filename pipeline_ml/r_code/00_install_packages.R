# =============================================================================
# 00_install_packages.R
# Install all packages required by the DC Property ML project.
# Run this once before executing any other script.
# =============================================================================

required_packages <- c(
  # Core data wrangling
  "dplyr",
  "tidyr",
  "readr",
  "purrr",
  "tibble",
  "rlang",

  # Config & project structure
  "yaml",
  "here",

  # Modelling
  "MASS",       # lda, qda, stepAIC
  "glmnet",     # Ridge, Lasso
  "pls",        # PCR, PLS
  "class",      # knn
  "rpart",      # Regression Trees
  "rpart.plot", # Tree visualisation
  "randomForest",
  "e1071",      # SVM
  "ranger",     # FAST random forest replacement

  # Diagnostics & variable selection
  "car",        # vif, outlierTest

  # Visualisation
  "ggplot2",
  "GGally",     # scatter matrix

  # Reporting
  "rmarkdown",
  "knitr",
  "kableExtra"
)

new_packages <- required_packages[
  !required_packages %in% installed.packages()[, "Package"]
]

if (length(new_packages) > 0) {
  message(sprintf("Installing %d missing packages: %s",
                  length(new_packages),
                  paste(new_packages, collapse = ", ")))
  install.packages(new_packages, repos = "https://cran.rstudio.com/")
} else {
  message("All required packages are already installed.")
}

source("R/regression_metrics.R")


message("Package check complete.")
