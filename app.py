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

# If your dataset uses different column names, you can rename them using a mapping.
# For example, many Ames Housing datasets use names like "Overall Qual" instead of "OverallQual".
column_mapping = {
    "Overall Qual": "OverallQual",
    "Gr Liv Area": "GrLivArea",
    "Year Built": "YearBuilt",
    "Total Bsmt SF": "TotalBsmtSF",
    "Garage Cars": "GarageCars"
}

# Rename columns if they exist in the DataFrame
df.rename(columns=column_mapping, inplace=True)

# Debug: Display renamed columns
st.write("Renamed Dataset Columns:", df.columns.tolist())

# Define the expected features and target column name.
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

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a regression model (using Linear Regression)
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

# Create input widgets for each feature using the training data statistics.
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

# Predict the sale price when the button is clicked.
if st.button("Predict Sale Price"):
    input_data = np.array([[overall_qual, gr_liv_area, year_built, total_bsmt_sf, garage_cars]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Sale Price: ${prediction[0]:,.2f}")
