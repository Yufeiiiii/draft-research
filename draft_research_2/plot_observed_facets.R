#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
})

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  normalizePath(getwd(), mustWork = TRUE)
}

script_dir <- get_script_dir()
data_path <- file.path(script_dir, "NHL_2009_2018_combined_cleaned.csv")
output_path <- file.path(script_dir, "results", "observed_values_faceted_scatter.png")

responses <- c("TOI_sum_min", "GP_sum", "PS_sum")
response_labels <- c(
  TOI_sum_min = "TOI",
  GP_sum = "GP",
  PS_sum = "PS"
)

nhl_data <- read.csv(data_path, stringsAsFactors = FALSE)
plot_data <- nhl_data[, c("Selection", responses)]
plot_data$Selection <- suppressWarnings(as.numeric(plot_data$Selection))

for (response in responses) {
  plot_data[[response]] <- suppressWarnings(as.numeric(plot_data[[response]]))
}

long_data <- reshape(
  plot_data,
  direction = "long",
  varying = responses,
  v.names = "observed_value",
  timevar = "response",
  times = responses
)

long_data <- long_data[!is.na(long_data$Selection) & !is.na(long_data$observed_value), ]
long_data$response <- factor(
  long_data$response,
  levels = responses,
  labels = unname(response_labels[responses])
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

plot_object <- ggplot(long_data, aes(x = Selection, y = observed_value)) +
  geom_point(alpha = 0.35, size = 1.2, color = "#4c78a8") +
  facet_wrap(~ response, scales = "free_y", nrow = 1) +
  labs(
    title = "observed values faceted by TOI, GP, PS",
    x = "Selection",
    y = "Observed Value"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )

ggsave(
  filename = output_path,
  plot = plot_object,
  width = 16,
  height = 5,
  dpi = 300
)

cat(sprintf("Saved plot to %s\n", normalizePath(output_path, mustWork = TRUE)))
