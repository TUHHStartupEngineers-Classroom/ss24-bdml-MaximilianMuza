data_dir <- "data/"
models_dir <- "models/"

library(h2o)
library(tidyverse)
library(readxl)
library(recipes)
library(rsample)

product_backorders_tbl <- read_csv(file = file.path(data_dir, "product_backorders.txt"))
product_backorders_tbl

set.seed(123)
split <- initial_split(product_backorders_tbl, prop = 3/4)

train_tbl <- training(split)
test_tbl  <- testing(split)

product_recipe_obj <- recipe(went_on_backorder ~., data = train_tbl) %>% 
  step_zv(all_predictors()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  # step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  prep()
product_recipe_obj

train_trafo_tbl <- bake(product_recipe_obj, new_data = train_tbl)
test_trafo_tbl  <- bake(product_recipe_obj, new_data = test_tbl)

h2o.init()

split_h2o <- h2o.splitFrame(as.h2o(train_trafo_tbl), ratios = c(3/4), seed = 123)
train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]
test_h2o  <- as.h2o(test_trafo_tbl)

y <- "went_on_backorder"
x <- setdiff(names(train_h2o), y)

automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame    = train_h2o,
  validation_frame  = valid_h2o,
  leaderboard_frame = test_h2o,
  max_runtime_secs  = 30,
  nfolds            = 5
)

slotNames(automl_models_h2o)
automl_models_h2o@leaderboard

leader_model <- automl_models_h2o@leader
leader_model

predictions <- h2o.predict(leader_model, newdata = test_h2o)
predictions

h2o.getModel(leader_model@model_id) %>%
  h2o.saveModel(path = models_dir)

