<<<<<<< HEAD
# survival_analysis
=======
# Survival Analysis

- Author: [Lina Faik](https://www.linkedin.com/in/lina-faik/)
- Date: February 2023

## Objective

This repository contains the code and notebooks used to train survival models to tackle real-world predictive problems. It was developed as an experimentation project to support the explanation blog posts around survival models. For more information, you can find the articles here:

1. Part I - Survival Analysis: Predict Time-To-Event With Machine Learning

   Practical Application to Customer Churn Prediction 

   (link coming soon)

2. Part II - Coming soon

<div class="alert alert-block alert-info"> You can find all my technical blog posts <a href = https://linafaik.medium.com/>here</a>. </div>

## Project Description

### Data

The data are from [Kaggle](https://www.kaggle.com/datasets/gsagar12/dspp1). They are related to a subscription-based digital product offering for financial advice that includes newsletters, webinars, and investment recommendations. More specifically, the data consist of the following information:

- Customer sign-up and cancellation dates at the product level
- Call center activity
- Customer demographics
- Product pricing info

### Code structure

```
datasets
├── customer_subscription # folder containing the initial datasets
│   ├── customer_cases.csv
│   ├── customer_info.csv
│   ├── customer_product.csv
│   ├── customer_info.csv
notebooks
├── 01_data_preprocessing.ipynb # clean and prepare data
├── 02_data_exploration.ipynb # explore the data
├── 03_modeling_survival_ml.ipynb # train multiple models
├── 04_evaluation.ipynb # evaluate models
outputs
├── data
│   ├── customer_subscription_clean.csv # pre-processed data
│   ├── scaler.pkl # fit scaler
│   ├── train_x.pkl # features used to train models
│   ├── train_y.pkl # target from the train set
│   ├── val_x.pkl # features used to evaluate models
│   ├── val_y.pkl # target from the validation set
├── models # folder containing the trained models
│   ├── cox_ph.pkl
│   ├── gradient_boosting.pkl
│   ├── ksvm.pkl
│   ├── random_survival_forests.pkl
│   ├── svm.pkl
├── model_scores.csv # model performance on the 5-cross test and validation set
src
├── train.py # general functions to train models           
├── train_survival_ml.py # functions to train survival models
├── evaluate.py # functions to evaluate models
```

## How to Use This Repository?

### Requirement

To code relies on the following libraries:

```
scikit-survival==0.19.0 
plotly==4.14.3
```

### Experiments

To run experiments, you need:

1. To download data from [Kaggle](https://www.kaggle.com/datasets/gsagar12/dspp1) and upload them in the directory `datasets/customer_subscription`.
2. Run the notebooks in the order suggested by their names. The associated code is in the `src` directory.
>>>>>>> master
