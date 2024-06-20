# Libraries
library(h2o)
library(tidyverse)
library(readxl)
library(recipes)
library(rsample)
library(skimr)
library(GGally)
library(tidyquant)
library(lime)
library(tools)
library(ggplot2)
# Data

# Load data definitions
definitions_raw_tbl   <- read_excel(file.path(data_dir, "data_definitions.xlsx"), sheet = 1, col_names = FALSE)
definitions_raw_tbl |> as_tibble() |> print()

# Load employee attrition
employee_attrition_tbl <- read_csv(file.path(data_dir, "datasets-1067-1925-WA_Fn-UseC_-HR-Employee-Attrition.txt"))
employee_attrition_tbl |> as_tibble() |> print()

# Make HR data readable
process_hr_data_readable <- function(data, definitions_tbl) {
  
  definitions_list <- definitions_tbl %>%
    fill(...1, .direction = "down") %>%
    filter(!is.na(...2)) %>%
    separate(...2, into = c("key", "value"), sep = " '", remove = TRUE) %>%
    rename(column_name = ...1) %>%
    mutate(key = as.numeric(key)) %>%
    mutate(value = value %>% str_replace(pattern = "'", replacement = "")) %>%
    split(.$column_name) %>%
    map(~ select(., -column_name)) %>%
    map(~ mutate(., value = as_factor(value))) 
  
  for (i in seq_along(definitions_list)) {
    list_name <- names(definitions_list)[i]
    colnames(definitions_list[[i]]) <- c(list_name, paste0(list_name, "_value"))
  }
  
  data_merged_tbl <- list(HR_Data = data) %>%
    append(definitions_list, after = 1) %>%
    reduce(left_join) %>%
    select(-one_of(names(definitions_list))) %>%
    set_names(str_replace_all(names(.), pattern = "_value", 
                              replacement = "")) %>%
    select(sort(names(.))) %>%
    mutate_if(is.character, as.factor) %>%
    mutate(
      
    )
  
  return(data_merged_tbl)
  
}

employee_attrition_readable_tbl <- process_hr_data_readable(employee_attrition_tbl, definitions_raw_tbl)

# Create training and testing dataset (Hold-out method)
set.seed(1234)
split <- initial_split(employee_attrition_readable_tbl, prop = 0.85)

train_tbl <- training(split)
test_tbl  <- testing(split)


# ML Preprocessing Recipe
recipe_obj <- recipe(Attrition ~ ., data = train_tbl) %>%
  step_zv(all_predictors()) %>%
  step_mutate_at(c("JobLevel", "StockOptionLevel"), fn = as.factor) %>% 
  prep()

recipe_obj

train_tbl <- bake(recipe_obj, new_data = train_tbl)
test_tbl  <- bake(recipe_obj, new_data = test_tbl)

# H2O Model
h2o.init()

split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.85), seed = 1234)
train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]
test_h2o  <- as.h2o(test_tbl)

# Set the target and predictors
y <- "Attrition"
x <- setdiff(names(train_h2o), y)

# Create AutoML models ONLY IF model doesn't exist yet
# automl_leaderboard <- h2o.automl(
#   x = x,
#   y = y,
#   training_frame    = train_h2o,
#   validation_frame  = valid_h2o,
#   leaderboard_frame = test_h2o,
#   max_runtime_secs  = 30,
#   nfolds            = 5 
# )
# automl_leader <- automl_leaderboard@leader

# Save leading model ONLY IF model doesn't exist yet
# h2o.getModel(automl_leader@model_id) %>%
#   h2o.saveModel(path = models_dir)

# Load model if it exists
automl_leader <- h2o.loadModel(file.path(models_dir, "StackedEnsemble_BestOfFamily_2_AutoML_1_20240619_143112"))
automl_leader

predictions_tbl <- automl_leader %>% 
  h2o.predict(newdata = as.h2o(test_tbl)) %>%
  as.tibble() %>%
  bind_cols(
    test_tbl %>%
      select(Attrition, EmployeeNumber)
  )
predictions_tbl

### LIME for Single Explanation
# Create test table for first employee
test_tbl %>%
  slice(1) %>%
  glimpse()

# Create explainer object
explainer <- train_tbl %>%
  select(-Attrition) %>%
  lime(
    model           = automl_leader,
    bin_continuous  = TRUE,
    n_bins          = 4,
    quantile_bins   = TRUE
  )
explainer

# Make explanation with explain()
explanation <- test_tbl %>%
  slice(1) %>%
  select(-Attrition) %>%
  lime::explain(
    explainer = explainer,
    # Because it is a binary classification model: 1
    n_labels   = 1,
    # number of features to be returned
    n_features = 8,
    # number of localized linear models
    n_permutations = 5000,
    # Let's start with 1
    kernel_width   = 4
  )
explanation

# Select relevant columns
explanation %>%
  as.tibble() %>%
  select(feature:prediction) 

# Plot features
g <- plot_features(explanation = explanation, ncol = 1)
g

### Adapt for multiple explanations
explanation <- test_tbl %>%
  slice(1:20) %>%
  select(-Attrition) %>%
  lime::explain(
    explainer = explainer,
    n_labels   = 1,
    n_features = 8,
    n_permutations = 5000,
    kernel_width   = 0.5
  )
explanation %>%
  as.tibble()

plot_features(explanation, ncol = 4)

plot_explanations(explanation)

### Code plot features method

# Create a label formatter
custom_labeller <- function(labels, multi_line = TRUE, sep = ': ') {
  names(labels) <- toTitleCase(
    gsub("_", " ", names(labels))
  )
  label_both(labels, multi_line, sep)
}

custom_feature_plot <- function(explanation, ncol) {
  # Define color palette
  colors = c(
    "Supports" = "blue",
    "Contradicts" = "red"
  )
  
  # Define fill color column that specifies the color of the row in the plot
  # based on its key
  explanation <- explanation %>%
    mutate(
      fill_color = ifelse(feature_weight > 0, "Supports", "Contradicts")
    )
  
  # Create description by combining case and label, and also
  # append feature description
  explanation$description <- with(explanation, factor(
    paste0(case, '_', label, format(feature_desc)),
    levels = paste0(case, '_', label, format(feature_desc))[order(abs(feature_weight))]
  ))
  
  # Format model R-squared and label probability to two decimal places
  explanation$explanation_fit <- format(explanation$model_r2, digits = 2)
  explanation$probability <- format(explanation$label_prob, digits = 2)
  
  # Convert label to a factor and order based on label probability in decreasing order
  explanation$label <- factor(explanation$label, 
                              levels = unique(explanation$label[order(explanation$label_prob, decreasing = TRUE)]))
  
  ggplot(explanation) +
    facet_wrap(
      ~ case + label + probability + explanation_fit,
      labeller = custom_labeller,
      scales = 'free_y',
      ncol = ncol
    ) +
    geom_col(aes(
      x = feature_weight,
      y = reorder(feature_desc, abs(feature_weight)),
      fill = fill_color
    )) +
    labs(
      title = "Feature Importance",
      x = "Feature Weight",
      y = "Feature Description",
      fill = "Support vs Contradiction"
    ) +
    theme_minimal()
}

custom_feature_plot(explanation = explanation, ncol=4)

### Code plot explanation method
custom_explanation_plot <- function(explanation) {
  custom_margin = 30
  
  plot <- ggplot(explanation, aes_(~case, ~feature_desc)) +
    geom_tile(aes_(fill = ~feature_weight)) +
    scale_x_discrete('Case', expand = c(0, 0)) +
    scale_y_discrete('Feature', expand = c(0, 0)) +
    scale_fill_gradient2('Feature Weight', low = 'red', mid = 'white', high = 'blue') +
    theme_minimal() +
    theme(
      panel.border = element_rect(fill = NA, colour = 'black', size = 1),
      panel.grid = element_blank(),
      plot.margin = margin(custom_margin, custom_margin, custom_margin, custom_margin),
      legend.position = 'bottom',
    )
  
  # Create subplots if for both label yes and no
  if (is.null(explanation$label)) {
    plot
  } else {
    plot + facet_wrap(~label)
  }
}
custom_explanation_plot(explanation = explanation)