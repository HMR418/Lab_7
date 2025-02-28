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

# Debug: Display the original column names
st.write("Original Dataset Columns:", df.columns.tolist())

# Remove any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Debug: Display cleaned column names
st.write("Cleaned Dataset Columns:", df.columns.tolist())

# Define the expected features and target column name.
# Adjust these names if your dataset uses different naming conventions.
features = ["OverallQual", "GrLivArea", "YearBuilt", "TotalBsmtSF", "GarageCars"]
target = "SalePrice"

# Check if all expected columns are present
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    st.error(f"Error: The following expected columns are missing in the dataset: {missing_cols}")
    st.stop()

# Drop rows with missing values for the selected features and target
df = df.dropna(subset=features + [target])

# Prepare feature matrix X and target vector y
X = df[features]
y = df[target]

# ---------------------------
# 2. Splitting the Data and Training the Model
# ---------------------------

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance on the test set
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

# Input widgets for each feature
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

# When the user clicks the "Predict" button, predict the sale price using the trained model.
if st.button("Predict Sale Price"):
    # Prepare the input data as a 2D numpy array
    input_data = np.array([[overall_qual, gr_liv_area, year_built, total_bsmt_sf, garage_cars]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Sale Price: ${prediction[0]:,.2f}")
