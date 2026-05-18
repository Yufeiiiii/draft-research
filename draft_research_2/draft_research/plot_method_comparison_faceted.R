#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(bnmr)
})

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  normalizePath(getwd(), mustWork = TRUE)
}

resolve_path <- function(candidates, label) {
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0) {
    stop(sprintf("[%s] Could not find any of:\n%s", label, paste(candidates, collapse = "\n")))
  }

  if (length(existing) > 1) {
    message(sprintf("[%s] Multiple candidate files found; using first project-local path:", label))
    message(sprintf("  %s", paste(normalizePath(existing, mustWork = TRUE), collapse = "\n  ")))
  }

  chosen <- normalizePath(existing[[1]], mustWork = TRUE)
  message(sprintf("[%s] Loaded file: %s", label, chosen))
  chosen
}

check_duplicate_picks <- function(df, pick_col, label) {
  duplicate_rows <- sum(duplicated(df[[pick_col]]))
  if (duplicate_rows > 0) {
    message(sprintf("[%s] Duplicate %s values found: %d", label, pick_col, duplicate_rows))
  } else {
    message(sprintf("[%s] No duplicate %s values found.", label, pick_col))
  }
  invisible(duplicate_rows)
}

collapse_curve_duplicates <- function(df, label) {
  check_duplicate_picks(df, "Pick", label)
  df %>%
    group_by(Pick) %>%
    summarise(fitted_value_raw = mean(fitted_value_raw, na.rm = TRUE), .groups = "drop") %>%
    arrange(Pick)
}

scale_curve_sum1000 <- function(curve_df, response_name, method_name, epsilon = 1e-8) {
  raw_values <- curve_df$fitted_value_raw
  min_value <- min(raw_values, na.rm = TRUE)
  raw_sum <- sum(raw_values, na.rm = TRUE)
  shift_value <- 0
  epsilon_added <- 0

  adjusted_values <- raw_values

  # If the fitted curve contains negative values, shift the full curve upward so
  # all adjusted values are nonnegative before applying the sum-to-1000 scaling.
  if (min_value < 0) {
    shift_value <- -min_value
    adjusted_values <- adjusted_values + shift_value
  }

  adjusted_sum <- sum(adjusted_values, na.rm = TRUE)

  # In edge cases such as an all-zero curve, or a curve that becomes all zero
  # after shifting, add a tiny constant so the scaling denominator is positive.
  if (adjusted_sum <= 0) {
    epsilon_added <- epsilon
    adjusted_values <- adjusted_values + epsilon
    adjusted_sum <- sum(adjusted_values, na.rm = TRUE)
  }

  if (adjusted_sum <= 0) {
    stop(sprintf("[%s - %s] Unable to scale fitted curve because the adjusted sum is still nonpositive.", response_name, method_name))
  }

  if (shift_value > 0) {
    message(sprintf("[%s - %s] Negative fitted values detected (min = %.6f); shifted upward by %.6f before scaling.", response_name, method_name, min_value, shift_value))
  } else {
    message(sprintf("[%s - %s] No shifting needed before scaling (raw sum = %.6f).", response_name, method_name, raw_sum))
  }

  if (epsilon_added > 0) {
    message(sprintf("[%s - %s] Added epsilon %.2e because adjusted sum was nonpositive before scaling.", response_name, method_name, epsilon_added))
  }

  curve_df %>%
    mutate(
      fitted_value_adjusted = adjusted_values,
      fitted_value_scaled = adjusted_values / adjusted_sum * 1000,
      shift_value = shift_value,
      epsilon_added = epsilon_added,
      raw_sum = raw_sum,
      adjusted_sum = adjusted_sum
    )
}

fit_loess_curve <- function(data, response_col, pick_col, span, family_name, label) {
  set.seed(1)
  ordered_data <- data %>% arrange(.data[[pick_col]])
  pick_grid <- data.frame(Pick = seq(min(ordered_data[[pick_col]], na.rm = TRUE), max(ordered_data[[pick_col]], na.rm = TRUE), by = 1))
  names(pick_grid) <- pick_col

  fit <- loess(
    formula = as.formula(sprintf("%s ~ %s", response_col, pick_col)),
    data = ordered_data,
    span = span,
    degree = 1,
    family = family_name,
    na.action = na.exclude,
    surface = "direct"
  )

  curve_df <- data.frame(
    Pick = pick_grid[[pick_col]],
    fitted_value_raw = as.numeric(predict(fit, newdata = pick_grid))
  ) %>%
    filter(!is.na(fitted_value_raw))

  collapse_curve_duplicates(curve_df, label)
}

fit_bnmr_curve <- function(data, response_col, pick_col, bnmr_args, label) {
  set.seed(1)
  fit <- do.call(
    bnmr::bnmr,
    c(
      list(
        y = data[[response_col]],
        x = data[[pick_col]]
      ),
      bnmr_args
    )
  )

  curve_df <- as.data.frame(fit$fitted) %>%
    transmute(
      Pick = as.integer(round(x)),
      fitted_value_raw = fitted
    ) %>%
    filter(!is.na(Pick), !is.na(fitted_value_raw))

  collapse_curve_duplicates(curve_df, label)
}

load_bspline_curve <- function(path, label) {
  curve_df <- read.csv(path, stringsAsFactors = FALSE) %>%
    transmute(
      Pick = as.integer(round(overall)),
      fitted_value_raw = raw_model_output
    ) %>%
    filter(!is.na(Pick), !is.na(fitted_value_raw))

  collapse_curve_duplicates(curve_df, label)
}

build_method_curve <- function(spec, method_name, curve_df) {
  scaled_curve <- scale_curve_sum1000(
    curve_df = curve_df,
    response_name = spec$response,
    method_name = method_name
  )

  scaled_curve %>%
    transmute(
      response = factor(spec$response, levels = response_levels),
      method = factor(method_name, levels = method_levels),
      Pick = Pick,
      fitted_value_raw = fitted_value_raw,
      fitted_value_scaled = fitted_value_scaled,
      shift_value = shift_value,
      epsilon_added = epsilon_added,
      raw_sum = raw_sum,
      adjusted_sum = adjusted_sum
    )
}

script_dir <- get_script_dir()
project_dir <- if (basename(script_dir) == "draft_research") {
  script_dir
} else if (dir.exists(file.path(script_dir, "draft_research"))) {
  normalizePath(file.path(script_dir, "draft_research"), mustWork = TRUE)
} else {
  stop("Could not locate the draft_research project directory from the script location.")
}
root_dir <- dirname(project_dir)
results_dir <- file.path(project_dir, "results")

response_levels <- c("GP_sum", "PS_sum", "TOI_sum_min")
method_levels <- c("Loess", "B-spline", "BNMR")

nhl_path <- resolve_path(
  c(
    file.path(project_dir, "NHL_2009_2018_combined_cleaned.csv"),
    file.path(root_dir, "NHL_2009_2018_combined_cleaned.csv")
  ),
  "NHL data"
)

nhl_data <- read.csv(nhl_path, stringsAsFactors = FALSE) %>%
  mutate(Selection = as.integer(Selection)) %>%
  filter(!is.na(Selection))

response_specs <- list(
  list(
    response = "GP_sum",
    source_data = nhl_data %>% filter(!is.na(GP_sum)),
    response_col = "GP_sum",
    pick_col = "Selection",
    loess_span = 0.45,
    loess_family = "gaussian",
    bnmr_args = list(
      M = 20,
      niter = 3000,
      nburn = 1500,
      nthin = 1,
      direction = "decreasing",
      priors = list(mu0 = 0.5, phi0 = 20, intsd = 0.4715247)
    ),
    bspline_path = resolve_path(
      c(file.path(project_dir, "results", "gp_sum", "draft_curve_gp_sum_lookup.csv")),
      "B-spline GP_sum"
    )
  ),
  list(
    response = "PS_sum",
    source_data = nhl_data %>% filter(!is.na(PS_sum)),
    response_col = "PS_sum",
    pick_col = "Selection",
    loess_span = 0.10,
    loess_family = "gaussian",
    bnmr_args = list(
      M = 10,
      niter = 3000,
      nburn = 1500,
      nthin = 1,
      direction = "decreasing",
      priors = list(mu0 = 0.5, phi0 = 10, intsd = 3.0740209)
    ),
    bspline_path = resolve_path(
      c(file.path(project_dir, "results", "ps_sum", "draft_curve_ps_sum_lookup.csv")),
      "B-spline PS_sum"
    )
  ),
  list(
    response = "TOI_sum_min",
    source_data = nhl_data %>% filter(!is.na(TOI_sum_min)),
    response_col = "TOI_sum_min",
    pick_col = "Selection",
    loess_span = 0.10,
    loess_family = "gaussian",
    bnmr_args = list(
      M = 20,
      niter = 3000,
      nburn = 1500,
      nthin = 1,
      direction = "decreasing",
      priors = list(mu0 = 0.5, phi0 = 20, intsd = 0.6232552)
    ),
    bspline_path = resolve_path(
      c(file.path(project_dir, "results", "toi_sum_min", "draft_curve_toi_sum_min_lookup.csv")),
      "B-spline TOI_sum_min"
    )
  )
)

curve_list <- lapply(response_specs, function(spec) {
  message(sprintf("[Loess - %s] Recomputing fitted curve from notebook settings.", spec$response))
  loess_curve <- fit_loess_curve(
    data = spec$source_data,
    response_col = spec$response_col,
    pick_col = spec$pick_col,
    span = spec$loess_span,
    family_name = spec$loess_family,
    label = sprintf("Loess %s", spec$response)
  )

  message(sprintf("[B-spline - %s] Reading lookup curve from CSV.", spec$response))
  bspline_curve <- load_bspline_curve(
    path = spec$bspline_path,
    label = sprintf("B-spline %s", spec$response)
  )

  message(sprintf("[BNMR - %s] Recomputing fitted curve from notebook settings.", spec$response))
  bnmr_curve <- fit_bnmr_curve(
    data = spec$source_data,
    response_col = spec$response_col,
    pick_col = spec$pick_col,
    bnmr_args = spec$bnmr_args,
    label = sprintf("BNMR %s", spec$response)
  )

  bind_rows(
    build_method_curve(spec, "Loess", loess_curve),
    build_method_curve(spec, "B-spline", bspline_curve),
    build_method_curve(spec, "BNMR", bnmr_curve)
  )
})

combined_curve_data <- bind_rows(curve_list) %>%
  arrange(response, method, Pick)

duplicate_combo_count <- combined_curve_data %>%
  count(response, method, Pick, name = "n") %>%
  filter(n > 1) %>%
  nrow()

if (duplicate_combo_count > 0) {
  message(sprintf("[Combined curves] Duplicate response-method-pick rows found after assembly: %d", duplicate_combo_count))
} else {
  message("[Combined curves] No duplicate response-method-pick rows found after assembly.")
}

combined_curve_csv <- file.path(results_dir, "draft_curve_method_comparison_faceted_sum1000.csv")
plot_output_path <- file.path(results_dir, "draft_curve_method_comparison_faceted_sum1000.png")

write.csv(combined_curve_data, combined_curve_csv, row.names = FALSE)
message(sprintf("[Output] Saved combined plotting CSV: %s", normalizePath(combined_curve_csv, mustWork = FALSE)))

# Raw observations are intentionally omitted here because the figure compares
# fitted curves on a sum-to-1000 scale, while the raw outcomes remain on their
# original units and would not share a directly interpretable y-axis.
comparison_plot <- ggplot(
  combined_curve_data,
  aes(x = Pick, y = fitted_value_scaled, color = method)
) +
  geom_line(linewidth = 1.05, alpha = 0.95) +
  facet_wrap(
    ~ response,
    ncol = 2,
    scales = "free_y",
    labeller = as_labeller(
      c(
        "GP_sum" = "GP",
        "PS_sum" = "PS",
        "TOI_sum_min" = "TOI"
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
    x = "Draft Pick / Overall Selection",
    y = "Scaled Fitted Value (curve sum = 1000)",
    color = "Method"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 10.5),
    strip.background = element_rect(fill = "grey95", color = "grey70"),
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position = "top",
    legend.title = element_text(face = "bold")
  )

ggsave(
  filename = plot_output_path,
  plot = comparison_plot,
  width = 14,
  height = 10,
  dpi = 300,
  bg = "white"
)

message(sprintf("[Output] Saved comparison plot: %s", normalizePath(plot_output_path, mustWork = FALSE)))
