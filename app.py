import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# 1. Data Loading and Preprocessing
# ---------------------------

# Read the dataset from the local file in the GitHub repository
df = pd.read_excel('AmesHousing.xlsx')

# For this example, we select a few key numerical features.
# You can adjust the feature list based on your dataset.
features = ["OverallQual", "GrLivArea", "YearBuilt", "TotalBsmtSF", "GarageCars"]
target = "SalePrice"

# Drop rows with missing values in selected features/target
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# ---------------------------
# 2. Splitting the Data and Training the Model
# ---------------------------

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a regression model (using Linear Regression in this example)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance on the test set (optional)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---------------------------
# 3. Streamlit Web Application
# ---------------------------

st.title("Ames Housing Price Prediction")
st.write("This app predicts the sale price of a house based on selected features.")

st.subheader("Model Performance on Test Set")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

st.subheader("Enter House Details:")

# Create input widgets for the selected features.
overall_qual = st.slider(
    "Overall Quality", 
    int(X["OverallQual"].min()), 
    int(X["OverallQual"].max()), 
    int(X["OverallQual"].median())
)
gr_liv_area = st.number_input(
    "Above Ground Living Area (sq ft)", 
    float(X["GrLivArea"].min()), 
    float(X["GrLivArea"].max()), 
    float(X["GrLivArea"].median())
)
year_built = st.number_input(
    "Year Built", 
    int(X["YearBuilt"].min()), 
    int(X["YearBuilt"].max()), 
    int(X["YearBuilt"].median())
)
total_bsmt_sf = st.number_input(
    "Total Basement Area (sq ft)", 
    float(X["TotalBsmtSF"].min()), 
    float(X["TotalBsmtSF"].max()), 
    float(X["TotalBsmtSF"].median())
)
garage_cars = st.slider(
    "Number of Cars in Garage", 
    int(X["GarageCars"].min()), 
    int(X["GarageCars"].max()), 
    int(X["GarageCars"].median())
)

# When the user clicks the "Predict" button, use the trained model to predict SalePrice.
if st.button("Predict Sale Price"):
    # Prepare the input data as a 2D numpy array
    input_data = np.array([[overall_qual, gr_liv_area, year_built, total_bsmt_sf, garage_cars]])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Sale Price: ${prediction[0]:,.2f}")
