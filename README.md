# Bank Predictive Model on Product Sales
<img src="/Images/Cover Photo.jpg"/>

## Introduction
The ability of banks to be able to know the probability of a client to choose a certain product that they are marketing is a key metric for business development.

This enables the bank to know which features or characteristics of the client influence whether the client will buy the said product.

## Objective
This project seeks to develop a machine learning model that predicts if the customer will buy the marketed bank product or not. 

## Uses Cases

This model (Concept behind the model )can be used by banks to test if a client will buy a specific product or not

## Data
The data has been obtained from UCI repository

It contains data collected from a Portugal bank clients on various columns and whether they purchased the product or not

The data contains 45,200 rows and 17 columns

## Metrics

We will use the F1 score to evaluate the model performance

## Insights

### Education by Client Subscription

<img src="/Images/education distribution.png"/>


### Job Category by Client subscription
<img src="/Images/Job distribution.png"/>

## Model 
First built a Regression model which attained F1 score of 0.7

Random Forest Model was built which attained an F1 Score of 0.94

This Score was obtained after hyper parameter tuning

The Confusion matrix is below

<img src="/Images/confusion matrix.png"/>

The Random Forest model was choosen for deployment

## Feature Importance

The following Features were the most important for model classificationm (Whether the individual will take the Bank product or not)

<img src="/Images/Feature Importance.png"/>

## Deployment

The Random Forest model weights were saved and deployed to streamlit for interface as seen below

<img src="/Images/Screenshot (12).png"/>

The Model can be tested here = http://192.168.100.9:8507

## Next Step

Plan on training the model only on the most important features then deploying

