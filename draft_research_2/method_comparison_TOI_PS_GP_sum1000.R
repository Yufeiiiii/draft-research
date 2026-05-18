#!/usr/bin/env Rscript

if (!grepl("^4\\.3\\.", getRversion())) {
  stop(
    "This script must be run with R 4.3 because the modified BNMR package is installed under R 4.3. ",
    "Current R version is: ", R.version.string
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(tidyr)
  library(bnmr)
})

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE)))
  }
  normalizePath(getwd(), mustWork = TRUE)
}

find_project_root <- function() {
  candidates <- unique(c(get_script_dir(), normalizePath(getwd(), mustWork = TRUE)))

  for (candidate in candidates) {
    if (dir.exists(file.path(candidate, "draft_research"))) {
      return(normalizePath(candidate, mustWork = TRUE))
    }
    if (basename(candidate) == "draft_research") {
      return(dirname(normalizePath(candidate, mustWork = TRUE)))
    }
  }

  stop("Could not locate the project root containing the draft_research folder.")
}

resolve_path <- function(candidates, label) {
  existing <- candidates[file.exists(candidates)]

  if (length(existing) == 0) {
    stop(
      sprintf(
        "[%s] Could not find any of these paths:\n%s",
        label,
        paste(candidates, collapse = "\n")
      )
    )
  }

  normalizePath(existing[[1]], mustWork = TRUE)
}

identify_column <- function(df, candidates, label, file_label) {
  matches <- intersect(candidates, names(df))

  if (length(matches) == 0) {
    stop(
      sprintf(
        "[%s] Could not identify %s column. Available columns: %s",
        file_label,
        label,
        paste(names(df), collapse = ", ")
      )
    )
  }

  matches[[1]]
}

complete_curve_to_224 <- function(curve_df,
                                  response,
                                  method,
                                  selection_grid,
                                  observed_max,
                                  interpolate_within_observed = FALSE,
                                  carry_forward_beyond_observed = FALSE) {
  completed <- selection_grid %>%
    left_join(curve_df %>% arrange(Selection), by = "Selection")

  within_observed <- completed$Selection <= observed_max
  beyond_observed <- completed$Selection > observed_max

  if (interpolate_within_observed) {
    observed_points <- completed %>%
      filter(!is.na(fitted_raw))

    if (nrow(observed_points) < 2) {
      stop(
        sprintf(
          "[%s - %s] Need at least two non-missing fitted values to interpolate.",
          response,
          method
        )
      )
    }

    approx_values <- approx(
      x = observed_points$Selection,
      y = observed_points$fitted_raw,
      xout = completed$Selection[within_observed],
      method = "linear",
      rule = 1
    )$y

    missing_within <- within_observed & is.na(completed$fitted_raw)
    completed$fitted_raw[missing_within] <- approx_values[is.na(completed$fitted_raw[within_observed])]
  }

  if (carry_forward_beyond_observed && any(beyond_observed)) {
    observed_nonmissing <- completed %>%
      filter(Selection <= observed_max, !is.na(fitted_raw)) %>%
      arrange(Selection)

    if (nrow(observed_nonmissing) == 0) {
      stop(
        sprintf(
          "[%s - %s] Cannot extend beyond observed range because no fitted values are available.",
          response,
          method
        )
      )
    }

    last_value <- observed_nonmissing$fitted_raw[[nrow(observed_nonmissing)]]
    completed$fitted_raw[beyond_observed & is.na(completed$fitted_raw)] <- last_value
  }

  if (any(is.na(completed$fitted_raw[within_observed]))) {
    stop(
      sprintf(
        "[%s - %s] Missing fitted values remain within the observed selection range 1:%d.",
        response,
        method,
        observed_max
      )
    )
  }

  if (any(is.na(completed$fitted_raw))) {
    missing_selections <- completed$Selection[is.na(completed$fitted_raw)]
    stop(
      sprintf(
        "[%s - %s] Missing fitted values remain after completion at selections: %s",
        response,
        method,
        paste(missing_selections, collapse = ", ")
      )
    )
  }

  completed
}

fit_loess_curve <- function(data, response, span, selection_grid) {
  set.seed(1)

  observed_max <- max(data$Selection, na.rm = TRUE)

  loess_fit <- loess(
    formula = as.formula(sprintf("%s ~ Selection", response)),
    data = data,
    span = span,
    degree = 1,
    family = "gaussian",
    na.action = na.exclude,
    surface = "direct"
  )

  raw_predictions <- predict(loess_fit, newdata = selection_grid)

  curve_df <- tibble(
    Selection = selection_grid$Selection,
    fitted_raw = as.numeric(raw_predictions)
  )

  # Base loess can return NA when the prediction grid extends beyond the
  # observed maximum pick. To keep the plotting grid consistent up to 224,
  # carry the last available fitted value forward only for those tail picks.
  completed <- complete_curve_to_224(
    curve_df = curve_df,
    response = response,
    method = "Loess",
    selection_grid = selection_grid,
    observed_max = observed_max,
    interpolate_within_observed = FALSE,
    carry_forward_beyond_observed = TRUE
  )

  completed %>%
    mutate(Response = response, Method = "Loess")
}

extract_bnmr_curve <- function(data, response, bnmr_args, selection_grid) {
  set.seed(1)

  observed_max <- max(data$Selection, na.rm = TRUE)

  bnmr_fit <- do.call(
    bnmr::bnmr,
    c(
      list(
        y = data[[response]],
        x = data$Selection
      ),
      bnmr_args
    )
  )

  use_predict <- exists("predict.bnmr", where = asNamespace("bnmr"), inherits = FALSE)

  if (use_predict) {
    predicted <- tryCatch(
      predict(bnmr_fit, newdata = selection_grid$Selection, interval = "none"),
      error = function(e) NULL
    )

    if (!is.null(predicted)) {
      curve_df <- as_tibble(predicted) %>%
        transmute(
          Selection = as.integer(round(x)),
          fitted_raw = as.numeric(fitted)
        )

      # predict.bnmr() is reliable on a new x grid within the observed range.
      # For selections beyond the observed maximum, it returns NA, so we extend
      # the curve to pick 224 by carrying forward the last in-range fitted value.
      completed <- complete_curve_to_224(
        curve_df = curve_df,
        response = response,
        method = "BNMR",
        selection_grid = selection_grid,
        observed_max = observed_max,
        interpolate_within_observed = FALSE,
        carry_forward_beyond_observed = TRUE
      )

      return(completed %>% mutate(Response = response, Method = "BNMR"))
    }
  }

  # Fallback if predict.bnmr() is unavailable or errors:
  # use fit$fitted, average duplicate selections, interpolate missing values
  # inside the observed range, and then carry the final fitted value through 224.
  curve_df <- as_tibble(bnmr_fit$fitted) %>%
    transmute(
      Selection = as.integer(round(x)),
      fitted_raw = as.numeric(fitted)
    ) %>%
    filter(!is.na(Selection), !is.na(fitted_raw)) %>%
    group_by(Selection) %>%
    summarise(fitted_raw = mean(fitted_raw), .groups = "drop") %>%
    arrange(Selection)

  completed <- complete_curve_to_224(
    curve_df = curve_df,
    response = response,
    method = "BNMR",
    selection_grid = selection_grid,
    observed_max = observed_max,
    interpolate_within_observed = TRUE,
    carry_forward_beyond_observed = TRUE
  )

  completed %>%
    mutate(Response = response, Method = "BNMR")
}

read_bspline_curve <- function(path, response, selection_grid) {
  lookup_df <- readr::read_csv(path, show_col_types = FALSE)

  x_col <- identify_column(
    lookup_df,
    candidates = c("Selection", "pick", "Pick", "x", "DraftOv", "overall"),
    label = "selection",
    file_label = response
  )

  fitted_col <- identify_column(
    lookup_df,
    candidates = c("raw_model_output", "fitted", "mean", "pred", "prediction", "expected_draft_value"),
    label = "fitted-value",
    file_label = response
  )

  curve_df <- lookup_df %>%
    transmute(
      Selection = as.integer(round(.data[[x_col]])),
      fitted_raw = as.numeric(.data[[fitted_col]])
    ) %>%
    filter(!is.na(Selection), !is.na(fitted_raw)) %>%
    group_by(Selection) %>%
    summarise(fitted_raw = mean(fitted_raw), .groups = "drop") %>%
    arrange(Selection)

  completed <- complete_curve_to_224(
    curve_df = curve_df,
    response = response,
    method = "B-spline",
    selection_grid = selection_grid,
    observed_max = max(selection_grid$Selection),
    interpolate_within_observed = FALSE,
    carry_forward_beyond_observed = FALSE
  )

  completed %>%
    mutate(Response = response, Method = "B-spline")
}

scale_sum1000 <- function(df, epsilon = 1e-8) {
  stopifnot(all(c("Selection", "Response", "Method", "fitted_raw") %in% names(df)))

  fitted_raw <- df$fitted_raw

  if (any(!is.finite(fitted_raw))) {
    stop(sprintf("[%s - %s] Non-finite fitted values detected before scaling.", df$Response[[1]], df$Method[[1]]))
  }

  raw_sum <- sum(fitted_raw)
  shift_applied <- any(fitted_raw < 0)
  shift_value <- if (shift_applied) -min(fitted_raw) else 0

  if (!shift_applied) {
    if (raw_sum <= 0 || !is.finite(raw_sum)) {
      stop(
        sprintf(
          "[%s - %s] Sum of nonnegative fitted values is not positive/finite: %s",
          df$Response[[1]],
          df$Method[[1]],
          raw_sum
        )
      )
    }

    fitted_adjusted <- fitted_raw
  } else {
    fitted_adjusted <- fitted_raw + shift_value

    if (any(fitted_adjusted <= 0)) {
      fitted_adjusted <- fitted_adjusted + epsilon
    }

    adjusted_sum <- sum(fitted_adjusted)

    if (adjusted_sum <= 0 || !is.finite(adjusted_sum)) {
      stop(
        sprintf(
          "[%s - %s] Sum of shifted fitted values is not positive/finite after applying shift %.8f.",
          df$Response[[1]],
          df$Method[[1]],
          shift_value
        )
      )
    }
  }

  adjusted_sum <- sum(fitted_adjusted)

  if (adjusted_sum <= 0 || !is.finite(adjusted_sum)) {
    stop(
      sprintf(
        "[%s - %s] Adjusted fitted values could not be scaled because their sum is %s.",
        df$Response[[1]],
        df$Method[[1]],
        adjusted_sum
      )
    )
  }

  df %>%
    mutate(
      fitted_adjusted = fitted_adjusted,
      scaled_value = fitted_adjusted / adjusted_sum * 1000,
      shift_applied = shift_applied,
      shift_value = shift_value
    ) %>%
    select(
      Selection,
      Response,
      Method,
      fitted_raw,
      fitted_adjusted,
      scaled_value,
      shift_applied,
      shift_value
    )
}

project_root <- find_project_root()
analysis_dir <- file.path(project_root, "draft_research")
results_dir <- file.path(analysis_dir, "results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

message("Running with: ", R.version.string)
message("Project root: ", project_root)
message("Analysis dir: ", analysis_dir)

nhl_path <- resolve_path(
  c(
    file.path(project_root, "NHL_2009_2018_combined_cleaned.csv"),
    file.path(analysis_dir, "NHL_2009_2018_combined_cleaned.csv")
  ),
  "NHL dataset"
)

nhl_data <- readr::read_csv(nhl_path, show_col_types = FALSE) %>%
  mutate(
    Selection = as.integer(Selection),
    TOI_sum_min = as.numeric(TOI_sum_min),
    PS_sum = as.numeric(PS_sum),
    GP_sum = as.numeric(GP_sum)
  )

required_cols <- c("Selection", "TOI_sum_min", "PS_sum", "GP_sum")
missing_cols <- setdiff(required_cols, names(nhl_data))
if (length(missing_cols) > 0) {
  stop("Missing required columns in NHL dataset: ", paste(missing_cols, collapse = ", "))
}

if (any(is.na(nhl_data[, required_cols]))) {
  stop("The main NHL dataset contains NA values in required columns after reading.")
}

selection_grid <- tibble(Selection = 1:224)

response_specs <- list(
  list(
    response = "TOI_sum_min",
    loess_span = 0.10,
    bnmr_args = list(
      M = 20,
      niter = 3000,
      nburn = 1500,
      nthin = 1,
      direction = "decreasing",
      priors = list(mu0 = 0.5, phi0 = 20, intsd = 0.6232552)
    ),
    bspline_path = resolve_path(
      c(file.path(analysis_dir, "results", "toi_sum_min", "draft_curve_toi_sum_min_lookup_extended_224.csv")),
      "TOI_sum_min B-spline lookup"
    )
  ),
  list(
    response = "PS_sum",
    loess_span = 0.10,
    bnmr_args = list(
      M = 10,
      niter = 3000,
      nburn = 1500,
      nthin = 1,
      direction = "decreasing",
      priors = list(mu0 = 0.5, phi0 = 10, intsd = 3.0740209)
    ),
    bspline_path = resolve_path(
      c(file.path(analysis_dir, "results", "ps_sum", "draft_curve_ps_sum_lookup_extended_224.csv")),
      "PS_sum B-spline lookup"
    )
  ),
  list(
    response = "GP_sum",
    loess_span = 0.45,
    bnmr_args = list(
      M = 20,
      niter = 3000,
      nburn = 1500,
      nthin = 1,
      direction = "decreasing",
      priors = list(mu0 = 0.5, phi0 = 20, intsd = 0.4715247)
    ),
    bspline_path = resolve_path(
      c(file.path(analysis_dir, "results", "gp_sum", "draft_curve_gp_sum_lookup_extended_224.csv")),
      "GP_sum B-spline lookup"
    )
  )
)

combined_curves <- bind_rows(lapply(response_specs, function(spec) {
  response_data <- nhl_data %>%
    select(Selection, all_of(spec$response)) %>%
    rename(ResponseValue = all_of(spec$response)) %>%
    arrange(Selection)

  loess_curve <- fit_loess_curve(
    data = response_data %>% rename(!!spec$response := ResponseValue),
    response = spec$response,
    span = spec$loess_span,
    selection_grid = selection_grid
  )

  bnmr_curve <- extract_bnmr_curve(
    data = response_data %>% rename(!!spec$response := ResponseValue),
    response = spec$response,
    bnmr_args = spec$bnmr_args,
    selection_grid = selection_grid
  )

  bspline_curve <- read_bspline_curve(
    path = spec$bspline_path,
    response = spec$response,
    selection_grid = selection_grid
  )

  bind_rows(loess_curve, bnmr_curve, bspline_curve) %>%
    group_split(Method, .keep = TRUE) %>%
    lapply(scale_sum1000) %>%
    bind_rows()
}))

combined_curves <- combined_curves %>%
  mutate(
    Response = factor(Response, levels = c("TOI_sum_min", "PS_sum", "GP_sum")),
    Method = factor(Method, levels = c("Loess", "B-spline", "BNMR"))
  ) %>%
  arrange(Response, Method, Selection)

coverage_summary <- combined_curves %>%
  group_by(Response, Method) %>%
  summarise(
    min_selection = min(Selection),
    max_selection = max(Selection),
    n = n(),
    sum_scaled = sum(scaled_value),
    min_raw = min(fitted_raw),
    min_adjusted = min(fitted_adjusted),
    shift_applied = any(shift_applied),
    .groups = "drop"
  )

message("\nCoverage summary:")
print(coverage_summary, width = Inf)

final_summary <- combined_curves %>%
  group_by(Response, Method) %>%
  summarise(
    min_selection = min(Selection),
    max_selection = max(Selection),
    n_rows = n(),
    raw_min_fitted = min(fitted_raw),
    shift_applied = any(shift_applied),
    sum_scaled = sum(scaled_value),
    .groups = "drop"
  )

message("\nFinal summary:")
print(final_summary, width = Inf)

output_csv <- file.path(results_dir, "method_comparison_TOI_PS_GP_sum1000.csv")
output_plot <- file.path(results_dir, "method_comparison_TOI_PS_GP_sum1000.png")

readr::write_csv(combined_curves, output_csv)

comparison_plot <- ggplot(
  combined_curves,
  aes(x = Selection, y = scaled_value, color = Method, linetype = Method)
) +
  geom_line(linewidth = 1.1) +
  facet_wrap(
    ~ Response,
    scales = "free_y",
    ncol = 1,
    labeller = as_labeller(
      c(
        "TOI_sum_min" = "TOI",
        "PS_sum" = "PS",
        "GP_sum" = "GP"
      )
    )
  ) +
  scale_color_manual(
    values = c(
      "Loess" = "#1b9e77",
      "B-spline" = "#d95f02",
      "BNMR" = "#7570b3"
    )
  ) +
  labs(
    title = "Comparison of Fitted NHL Draft Curves",
    subtitle = "Loess, monotonic Bayesian B-spline, and BNMR; each curve scaled to sum 1000 within response and method",
    x = "Draft pick / overall selection",
    y = "Scaled fitted value, sum = 1000",
    color = "Method",
    linetype = "Method"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave(
  filename = output_plot,
  plot = comparison_plot,
  width = 11,
  height = 12,
  dpi = 300,
  bg = "white"
)

message("\nSaved combined curve data to: ", output_csv)
message("Saved comparison plot to: ", output_plot)
