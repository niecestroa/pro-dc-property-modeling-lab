# =============================================================================
# 02_load_data.R
# Load the raw DC Properties CSV, select relevant columns, apply row-level
# filters identical to the original paper, and fill NAs.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

#' Load and minimally clean the raw DC Properties dataset
#'
#' Steps mirror the original paper:
#'   1. Read CSV
#'   2. Compute age columns from year fields
#'   3. Select the 21 analysis columns
#'   4. Replace all NAs with 0
#'   5. Filter: Price_10K > 1.0  (i.e. sale price > $10,000)
#'              ROOMS <= 40       (removes implausible records)
#'
#' @param cfg    Config list from load_config().
#' @param verbose Print progress messages? Default TRUE.
#' @return A tibble of the filtered dataset (≈57,610 rows × 21 cols).
load_raw_data <- function(cfg, verbose = TRUE) {

  path <- here::here(cfg$paths$raw_data)

  if (!file.exists(path)) {
    stop(sprintf(
      "[load_raw_data] Raw data not found at: %s\n  Place DC_Properties.csv in data/raw/",
      path
    ))
  }

  if (verbose) message("[load_raw_data] Reading CSV ...")
  raw <- readr::read_csv(path, show_col_types = FALSE)

  if (verbose) message(sprintf("[load_raw_data] Raw shape: %d rows x %d cols",
                               nrow(raw), ncol(raw)))

  # ---- 1. Derive age columns -------------------------------------------------
  current_year <- 2019L   # paper written Dec 2019; keeps reproducibility

  raw <- raw %>%
    dplyr::mutate(
      Price_10K   = PRICE / 10000,
      AYB_age     = current_year - AYB,
      EYB_age     = current_year - EYB,
      Remodel_age = current_year - SALEDATE_year(SALEDATE)
    )

  # ---- 2. Select analysis columns -------------------------------------------
  keep_cols <- c(
    "Price_10K",
    cfg$columns$numeric,          # BATHRM ROOMS BEDRM KITCHENS FIREPLACES
                                  # STORIES AYB_age EYB_age Remodel_age
                                  # LATITUDE LONGITUDE
    cfg$columns$categorical       # CNDTN GRADE WARD AC QUALIFIED
  )

  # Keep only columns that actually exist in the CSV (defensive)
  keep_cols <- intersect(keep_cols, colnames(raw))
  df <- raw %>% dplyr::select(dplyr::all_of(keep_cols))

  # ---- 3. Fill NAs by column type -------------------------------------------
  na_before <- sum(is.na(df))
  
  df <- df %>%
    mutate(
      across(where(is.numeric),
             ~ tidyr::replace_na(.x, cfg$cleaning$na_fill_value)),
      across(where(is.character),
             ~ tidyr::replace_na(.x, "Unknown")),
      across(where(is.factor),
             ~ forcats::fct_explicit_na(.x, na_level = "Unknown"))
    )
  
  if (verbose) message(sprintf(
    "[load_raw_data] Filled %d NA values (numeric→%s, categorical→'Unknown')",
    na_before, cfg$cleaning$na_fill_value
  ))
  

  # ---- 4. Row-level filters --------------------------------------------------
  n_before <- nrow(df)
  df <- df %>%
    dplyr::filter(
      Price_10K > cfg$cleaning$min_price_10k,
      ROOMS     <= cfg$cleaning$max_rooms
    )
  n_dropped <- n_before - nrow(df)
  if (verbose) message(sprintf("[load_raw_data] Dropped %d rows (price/rooms filters). Final: %d rows.",
                               n_dropped, nrow(df)))

  # ---- 5. Rename to paper-friendly names ------------------------------------
  df <- df %>%
    dplyr::rename(
      Bathrooms   = BATHRM,
      Rooms       = ROOMS,
      Bedrooms    = BEDRM,
      Kitchens    = KITCHENS,
      Fireplaces  = FIREPLACES,
      Stories     = STORIES,
      Latitude    = LATITUDE,
      Longitude   = LONGITUDE,
      Condition   = CNDTN,
      Grade       = GRADE,
      Ward        = WARD,
      Qualified   = QUALIFIED
    )

  # ---- 6. Coerce categoricals ------------------------------------------------
  cat_cols <- c("Condition", "Grade", "Ward", "AC", "Qualified")
  cat_cols <- intersect(cat_cols, colnames(df))
  df <- df %>%
    dplyr::mutate(dplyr::across(dplyr::all_of(cat_cols), as.factor))

  if (verbose) message(sprintf("[load_raw_data] Done. Final dataset: %d rows x %d cols",
                               nrow(df), ncol(df)))
  df
}

# Helper: extract year from SALEDATE (handles character or Date types)
SALEDATE_year <- function(x) {
  as.integer(format(as.Date(as.character(x), tryFormats = c("%Y-%m-%d", "%m/%d/%Y")), "%Y"))
}

#' Save cleaned dataset to disk as an RDS file
#'
#' @param df      Cleaned tibble.
#' @param cfg     Config list.
save_processed_data <- function(df, cfg) {
  out_path <- here::here(cfg$paths$processed_data)
  dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(df, out_path)
  message(sprintf("[save_processed_data] Saved to %s", out_path))
}
