# House Price Prediction using Linear Regression

## Project Overview

This project implements a Linear Regression based machine learning model to predict house prices using selected features from a housing dataset.

The model learns the relationship between house characteristics and their sale prices and uses this relationship to predict prices for new houses.

## Objective

The objective of this project is to develop a machine learning model that can predict the sale price of a house based on important property features.

The model uses features such as:

- GrLivArea - Above-ground living area
- Bedroom - Number of bedrooms
- FullBath - Number of full bathrooms

## Dataset

The project uses a housing dataset containing information about residential properties and their sale prices.

The target variable is:

```text
SalePrice

SalePrice

The selected input features are:

GrLivArea
Bedroom
FullBath

The dataset is used locally and is not included in the repository if it is large or provided separately.

Machine Learning Algorithm
Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical value.

In this project, Linear Regression learns the relationship between the selected house features and the house sale price.

The model can then use these learned relationships to predict the price of an unseen house.

Methodology

The project follows these steps:

Load the housing dataset
Explore the dataset
Select relevant features
Select the target variable
Handle the required data preprocessing
Split the dataset into training and testing sets
Train the Linear Regression model
Make predictions on the test data
Evaluate the model
Visualize the prediction results
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Model Evaluation

The trained model is evaluated using regression performance metrics such as:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score

These metrics help measure how accurately the model predicts house prices.

Project Structure
SCT_ML_1/
│
├── house_price_prediction.py
├── dataset.csv
└── README.md

The dataset file may be excluded from GitHub if required because of its size or licensing restrictions.

How to Run

Install the required libraries:

pip install pandas numpy scikit-learn matplotlib

Run the Python program:

python house_price_prediction.py

The program loads the dataset, trains the Linear Regression model, predicts house prices, and displays the evaluation results.

Result

The trained Linear Regression model predicts house prices based on the selected property features.

The model performance is evaluated using regression metrics and visualization of the prediction results.

Internship Task

SkillCraft Technology — Machine Learning Internship

Task 1: Implement a Linear Regression model to predict house prices based on selected features.