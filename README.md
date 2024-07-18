
# Museum Accreditation Status Prediction

## Objective

The objective of this project is to predict the accreditation status of a museum based on various features such as Governance, subject matter of the museum, and other relevant factors. This project utilizes workflow sets to develop accurate models to assist in the evaluation process of museums' accreditation.





## Project Overview
This project involves several key steps:
#### EDA
- Conducted a thorough EDA to understand the distribution, relationships, and patterns in the data.

#### Data Preprocessing:
- Utilized packages in the tidymodels suite to split the data into training and testing sets.
- Performed feature engineering, which included using generalized effect encodings for categorical variables with many levels.

#### Model Building and Tuning:

-Developed three different models: XGBoost, MARS (Multivariate Adaptive Regression Splines), and Random Forest.
- Performed model tuning using tune_race_anova to find the optimal hyperparameters for each model..
#### Model Evaluation:

- Evaluated the performance of the three models using appropriate metrics.
- Selected the best performing model based on these evaluations.
#### Test Set Evaluation

- Evaluated the performance of the best model on the test data to ensure its generalizability.
#### Feature Importance Analysis

- Determined the variable importance of the features to understand their impact on the model's predictions.

## Libraries 
- Tidyverse
- Tidymodels
- vip
- finetune
- embed


