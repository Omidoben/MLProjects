library(tidyverse)

library(tidymodels)

museums <- read_csv("museums.csv")
View(museums)

glimpse(museums)

# The data has 4,191 rows with 35 columns

# Build a model that predicts the accreditation status of a museum (Accredited, Unaccredited)

museums_df <- museums %>% 
  select(museum_id, Accreditation, Latitude, Longitude, Governance, 
         Size, Subject_Matter, Year_opened, Year_closed,
         Area_Deprivation_index, Area_Geodemographic_group)

museums_df
glimpse(museums_df)

# EDA

museums_df %>% 
  count(Accreditation) %>% 
  mutate(prop = n / sum(n))

# 41% of the museums are accredited while 59% are not

## Does accreditation of museums vary with Governance?

top_Governance <- museums_df %>% 
  count(Governance) %>% 
  slice_max(n, n = 6) %>% 
  pull(Governance)

top_Governance

museums_df %>% 
  filter(Governance %in% top_Governance) %>% 
  count(Governance, Accreditation) %>% 
  ggplot(aes(Accreditation, n, fill = Accreditation)) +
  geom_col(position = "dodge") +
  facet_wrap(~Governance, scales = "free_y") +
  theme_bw() +
  theme(axis.text = element_text(color = "black"))

# Function to simplify the work

p1 <- function(var){
  top <- museums_df %>% 
    count({{var}}) %>% 
    slice_max(n, n = 6) %>% 
    pull({{var}})
  
  museums_df %>% 
    filter({{var}} %in% top) %>% 
    count({{var}}, Accreditation) %>% 
    ggplot(aes(x = Accreditation, y = n, fill = Accreditation)) +
    geom_col(position = "dodge") +
    facet_wrap(vars({{var}}), scales = "free_y") +
    theme_bw() +
    theme(axis.text = element_text(color = "black"))
}

p1(Governance)

# Most of the museums governed by the Government Local Authority, Independent National Trust,
# and University are Accredited while majority of those governed by Independent-Not for profit,
# Independent-Private, and Independent Unknown are unaccredited


museums_df %>% 
  count(Size, Accreditation)

## Does the subject matter of a museum affect the accreditation status

museums_df %>% 
  count(Subject_Matter, sort = TRUE)

# There are many levels of Subject matter, which might require the use of effect encodings to handle this high cardinality

museums_df %>% 
  count(Subject_Matter) %>% 
  slice_max(n, n = 6) 

p1(Subject_Matter)

# Majority of the museums whose subject matters are Arts Fine & Decorative arts, Local Histories,
# Mixed Encyclopaedic, War and Conflict Regiment are accredited, while those for Buildings-Houses-Medium_houses, 
# Transport-Trains_and_railways are unaccredited


ggplot(museums_df, aes(Latitude, Longitude, color = Accreditation)) +
  geom_point(alpha = 0.5)

# The accredited and unaccredited museums are evenly spread across the cities


## Fix the year opened and year closed columns

unique(museums_df$Year_closed)

museums_df <- museums_df %>% 
  mutate(Year_opened = parse_number(Year_opened),
         Closed = if_else(Year_closed == "9999:9999", "Open", "Closed")) %>% 
  select(-Year_closed)


## Explore the Area columns

museums_df %>% 
  count(Area_Geodemographic_group)

p1(Area_Geodemographic_group)

# Majority of the museums in Country Living, glish and Welsh countryside, Remoter coastal Living,
# Scottish country side, and Thriving rural are unaccredited, while museums in Larger towns and cities
# have approximately equal levels of accreditation

####################################################################################

## Feature Engineering

museums_df <- museums_df %>% 
  mutate_if(is.character, factor) %>% 
  mutate(museum_id = as.character(museum_id)) %>% 
  na.omit()


# Train and test sets

set.seed(123)

museums_splits <- initial_split(museums_df, strata = Accreditation)
museums_splits

museum_train <- training(museums_splits)
museum_train

museum_test <- testing(museums_splits)
museum_test

# Cross validation folds

set.seed(234)
museum_folds <- vfold_cv(museum_train, strata = Accreditation)
museum_folds

library(embed)

# The subject_matter feature has high cardinality, thus we use effect encodings 

# Create a recipe

museum_rec <- recipe(Accreditation ~ ., data = museum_train) %>% 
  update_role(museum_id, new_role = "id") %>% 
  step_lencode_glm(Subject_Matter, outcome = vars(Accreditation)) %>% 
  step_dummy(all_nominal_predictors())

museum_rec

prep(museum_rec) %>% bake(new_data = NULL) %>% View()


# Create model specifications

# xgboost model

xgb_spec <- 
  boost_tree(trees = 1000, mtry = tune(), min_n = tune(), learn_rate = tune(), 
             loss_reduction = tune(), sample_size = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

xgb_spec

# Random forest model

rf_spec <- 
  rand_forest(trees = 1000, min_n = tune(), mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

rf_spec

# Mars model

mars_spec <- 
  mars(num_terms = tune(), prod_degree = tune()) %>% 
  set_engine("earth") %>% 
  set_mode("classification")

mars_spec


## Create workflow sets

museum_wfs <- 
  workflow_set(preproc = list(rec = museum_rec),
               models = list(xgb = xgb_spec,
                             rf = rf_spec,
                             mars = mars_spec))

museum_wfs

museum_wfs <- museum_wfs %>% 
  mutate(wflow_id = str_replace_all(wflow_id, "rec_", ""))


## Tune the models using tune_race_anova

library(finetune)

doParallel::registerDoParallel()

ctrl = control_race(save_pred = TRUE,
                    verbose_elim = TRUE,
                    save_workflow = TRUE)

ctrl

museum_race <- museum_wfs %>% 
  workflow_map(
    "tune_race_anova",
    seed = 1234,
    resamples = museum_folds,
    grid = 15,
    control = ctrl
  )

museum_race

collect_metrics(museum_race)

museum_race %>% 
  rank_results() %>% 
  filter(.metric == "roc_auc")


autoplot(museum_race,
         rank_metric = "roc_auc",
         metric = "roc_auc",
         select_best = TRUE) +
  ggrepel::geom_text_repel(aes(label = wflow_id), nudge_x = 0.1, nudge_y = 1/600) +
  theme(legend.position = "none")

# Random forest is the best performing model

museum_race %>% 
  extract_workflow_set_result("rf") %>% 
  show_best(metric = "roc_auc")

best_results <- museum_race %>% 
  extract_workflow_set_result("rf") %>% 
  select_best(metric = "roc_auc")

## Fit the final model

final_museum_fit <- museum_race %>% 
  extract_workflow("rf") %>% 
  finalize_workflow(best_results) %>% 
  last_fit(museums_splits)

final_museum_fit

final_museum_fit %>% collect_metrics()

## collect predictions

final_museum_fit %>% 
  collect_predictions() %>% 
  conf_mat(Accreditation, .pred_class)

## roc curve

final_museum_fit %>% 
  collect_predictions() %>% 
  roc_curve(Accreditation, .pred_Accredited) %>% 
  autoplot()

## variable importance

library(vip)

final_museum_fit %>% 
  extract_fit_engine() %>% 
  vip()

# Subject matter, the feature we did effect encoding on, is the most important variable in predicting accreditation status


