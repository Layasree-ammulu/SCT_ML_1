# ============================================================
# TASK 01 - HOUSE PRICE PREDICTION
# SkillCraft Technology - Machine Learning Internship
# Algorithm: Linear Regression
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 3. DISPLAY FIRST 5 ROWS
# ============================================================

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. DISPLAY COLUMN NAMES
# ============================================================

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath"
]

target = "SalePrice"

print("\nMissing values:")

print(
    df[
        features + [target]
    ].isnull().sum()
)


# ============================================================
# 6. SELECT FEATURES AND TARGET
# ============================================================

# GrLivArea -> Square footage
# BedroomAbvGr -> Number of bedrooms
# FullBath -> Number of bathrooms
# SalePrice -> House price

X = df[features]

y = df[target]


print("\nSelected features:")
print(X.head())


print("\nTarget values:")
print(y.head())


# ============================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining data:", X_train.shape)

print("Testing data:", X_test.shape)


# ============================================================
# 8. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 9. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# ============================================================
# 10. DISPLAY MODEL COEFFICIENTS
# ============================================================

print("\nModel coefficients:")

for feature, coefficient in zip(
    features,
    model.coef_
):
    print(
        f"{feature}: {coefficient:.2f}"
    )


print(
    "Intercept:",
    round(model.intercept_, 2)
)


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 12. CREATE ACTUAL VS PREDICTED TABLE
# ============================================================

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})


print("\nActual vs Predicted prices:")

print(results.head(10))


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n================ MODEL EVALUATION ================")

print(
    f"Mean Absolute Error (MAE): {mae:.2f}"
)

print(
    f"Mean Squared Error (MSE): {mse:.2f}"
)

print(
    f"Root Mean Squared Error (RMSE): {rmse:.2f}"
)

print(
    f"R² Score: {r2:.4f}"
)


# ============================================================
# 14. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel(
    "Actual House Prices"
)

plt.ylabel(
    "Predicted House Prices"
)

plt.title(
    "Actual vs Predicted House Prices"
)


# Perfect prediction line

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()

# Save graph
plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 15. SQUARE FOOTAGE VS HOUSE PRICE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["GrLivArea"],
    df["SalePrice"]
)

plt.xlabel(
    "Square Footage"
)

plt.ylabel(
    "House Price"
)

plt.title(
    "Square Footage vs House Price"
)

plt.tight_layout()

# Save graph
plt.savefig(
    "square_footage_vs_price.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 16. BEDROOMS VS HOUSE PRICE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["BedroomAbvGr"],
    df["SalePrice"]
)

plt.xlabel(
    "Number of Bedrooms"
)

plt.ylabel(
    "House Price"
)

plt.title(
    "Bedrooms vs House Price"
)

plt.tight_layout()

# Save graph
plt.savefig(
    "bedrooms_vs_price.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 17. BATHROOMS VS HOUSE PRICE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["FullBath"],
    df["SalePrice"]
)

plt.xlabel(
    "Number of Bathrooms"
)

plt.ylabel(
    "House Price"
)

plt.title(
    "Bathrooms vs House Price"
)

plt.tight_layout()

# Save graph
plt.savefig(
    "bathrooms_vs_price.png",
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 18. PREDICT PRICE FOR A NEW HOUSE
# ============================================================

# Example house:
# Square footage = 2000
# Bedrooms = 3
# Bathrooms = 2

new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "FullBath": [2]
})


predicted_price = model.predict(
    new_house
)


print(
    "\n================ NEW HOUSE PREDICTION ================"
)

print(
    "Square footage: 2000"
)

print(
    "Bedrooms: 3"
)

print(
    "Bathrooms: 2"
)

print(
    f"Predicted house price: ${predicted_price[0]:,.2f}"
)


# ============================================================
# 19. SAVE ALL TEST PREDICTIONS
# ============================================================

results.to_csv(
    "house_price_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to house_price_predictions.csv"
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print(
    "\n================ PROJECT COMPLETED ================"
)

print(
    "Task 1 - House Price Prediction completed successfully!"
)

print(
    "All graphs have been saved in the current folder."
)