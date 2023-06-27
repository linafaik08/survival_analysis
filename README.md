# Survival Analysis

- Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
- Creation date: February 2023
- Last update: April 2023

## Objective

This repository contains the code and notebooks used to train survival models to tackle real-world predictive problems. It was developed as an experimentation project to support the explanation blog posts around survival models. For more information, you can find the articles here:

1. [Part I - Survival Analysis: Predict Time-To-Event With Machine Learning](https://towardsdatascience.com/survival-analysis-predict-time-to-event-with-machine-learning-part-i-ba52f9ab9a46)

   Practical Application to Customer Churn Prediction

2. [Part II - Survival Analysis: Leveraging Deep Learning for Time-to-Event Forecasting](https://towardsdatascience.com/survival-analysis-leveraging-deep-learning-for-time-to-event-forecasting-5c55bd4bb066)    
   
   Practical Application to Rehospitalization

<div class="alert alert-block alert-info"> You can find all my technical blog posts <a href = https://linafaik.medium.com/>here</a>. </div>

## Project Description

### Data

The project consists of two use cases. Each one is described in a different article.

The data used in part 1 is from [Kaggle](https://www.kaggle.com/datasets/gsagar12/dspp1). 
They are related to a subscription-based digital product offering for financial advice that includes newsletters, webinars, and investment recommendations. More specifically, the data consist of the following information:

- Customer sign-up and cancellation dates at the product level
- Call center activity
- Customer demographics
- Product pricing info

The data used in part 2 is from [Kaggle](https://www.kaggle.com/datasets/ashishsahani/hospital-admissions-data?select=HDHI+Admission+data.csv) and described in this [research paper](https://www.mdpi.com/2075-4418/12/2/241).
It was collected from patients admitted over a period of two years at Hero DMC Heart Institute in India.  
The data consists of information about the patient including:
- Demographics: age, gender, locality (rural or urban)
- Patient history: smoking, alcohol, diabetes mellitus, hypertension, etc.
- Lab results: hemoglobin, total lymphocyte count, platelets, glucose, urea, creatinine, etc.

### Code structure

```
datasets # folder containing the initial datasets
├── customer_subscription # used for the use case described in part 1
│   ├── customer_cases.csv
│   ├── customer_info.csv
│   ├── customer_product.csv
│   ├── customer_info.csv
├── hospitalisation # used for the use case described in part 2
│   ├── HDHI Admission data.csv
│   ├── HDHI Mortality data.csv
│   ├── HDHI Pollution data.csv
│   ├── table_headings.csv
notebooks
├── 01_data_preprocessing_customer_subscription.ipynb # clean and prepare data in part 1
├── 02_data_exploration_customer_subscription.ipynb # explore the data in part 1
├── 03_modeling_survival_ml_customer_subscription.ipynb # train multiple models in part 1
├── 04_evaluation_customer_subscription.ipynb # evaluate models in part 1
├── 11_data_preprocessing_customer_hospitalisation.ipynb # clean and prepare data in part 2
├── 12_data_exploration_customer_hospitalisation.ipynb # explore the data in part 1
├── 13_modeling_survival_ml_hospitalisation.ipynb # train multiple models in part 1
├── 14_evaluation_customer_hospitalisation.ipynb # evaluate models in part 1
outputs
├── data
│   ├── customer_subscription_clean.csv # pre-processed data in part 1
│   ├── hdhi_clean.csv # pre-processed data in part 2
│   ├── scaler.pkl # fitted scaler
│   ├── imputation_values.pkl # values used for importation
│   ├── train_x.pkl # features used to train models
│   ├── train_y.pkl # target from the train set
│   ├── val_x.pkl # features used to evaluate models
│   ├── val_y.pkl # target from the validation set
├── models # folder containing the trained models
├── model_scores.csv # model performance in part 1
├── model_scores_dl.csv # model performance in part 2
src
├── train.py # general functions to train models           
├── train_survival_ml.py # functions to train survival models
├── train_survival_deep.py # functions to train deep learning survival models
├── evaluate.py # functions to evaluate models
```

## How to Use This Repository?

### Requirement

To code relies on the following libraries:

```
scikit-survival==0.19.0 
plotly==4.14.3
torch==1.13.1
torchtuples==0.2.2
pycox==0.2.3
```

### Experiments

To run experiments, you need to run the notebooks in the order suggested by their names. 
The associated code is in the `src` directory.
