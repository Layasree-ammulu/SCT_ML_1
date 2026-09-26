# House Price Prediction using Linear Regression

## Project Overview

This project implements a Linear Regression based machine learning model to predict house prices using selected features from a housing dataset.

The model learns the relationship between house characteristics and their sale prices and uses this relationship to predict prices for new houses.

## Objective

The objective of this project is to develop a machine learning model that can predict the sale price of a house based on important property features.

The model uses the following features:

- **GrLivArea** - Above-ground living area
- **Bedroom** - Number of bedrooms
- **FullBath** - Number of full bathrooms

## Dataset

The project uses a housing dataset containing information about residential properties and their sale prices.

### Target Variable

SalePrice

### Input Features

- GrLivArea
- Bedroom
- FullBath

The dataset used for training is provided as `train.csv`.

## Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values.

In this project, Linear Regression learns the relationship between the selected house features and the house sale price.

The learned relationship is then used to predict prices for unseen houses.

## Methodology

The project follows these steps:

1. Load the housing dataset
2. Explore the dataset
3. Select relevant features
4. Select the target variable
5. Perform required data preprocessing
6. Split the dataset into training and testing sets
7. Train the Linear Regression model
8. Make predictions on the test data
9. Evaluate the model
10. Visualize the prediction results

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Model Evaluation

The trained model is evaluated using the following regression metrics:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

These metrics help measure the performance of the Linear Regression model.

## Visualizations

The project generates visualizations to understand the relationship between house features and prices and to compare actual prices with predicted prices.

Generated plots include:

- Square Footage vs Price
- Bedrooms vs Price
- Bathrooms vs Price
- Actual vs Predicted Prices

## Project Structure

```text
SCT_ML_1/
│
├── house_prediction.py
├── train.csv
├── house_price_predictions.csv
├── actual_vs_predicted.png
├── square_footage_vs_price.png
├── bedrooms_vs_price.png
├── bathrooms_vs_price.png
├── output.txt
├── requirements.txt
└── readme.md
```
## How to Run

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Run the Python program

```bash
python house_prediction.py
```

The program loads the dataset, trains the Linear Regression model, predicts house prices, evaluates the model, and generates visualization files.

## Result

The Linear Regression model predicts house prices based on selected property features.

The model performance is evaluated using MAE, MSE, and R² Score, along with visualizations comparing actual and predicted prices.

## Internship Task

**Organization:** SkillCraft Technology

**Program:** Machine Learning Internship

**Task 1:** Implement a Linear Regression model to predict house prices based on selected features.