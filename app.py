import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Housing Price Prediction App")

# -----------------------------
# Data Loading and Validation
# -----------------------------
try:
    # Read the dataset from the local file in the GitHub repository
    df = pd.read_excel('AmesHousing.xlsx')
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

if df.empty:
    st.error("The dataset is empty. Please check the file path and file contents.")
    st.stop()

# -----------------------------
# Data Imputation (Handling Missing Values)
# -----------------------------
# Identify numeric columns and fill missing values with the mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Optionally, if there are non-numeric columns you care about,
# you might fill missing values with a default value or mode.
# For now, we assume the features used for prediction are numeric.

# -----------------------------
# Data Preprocessing and Splitting
# -----------------------------
# Define the features and target variable
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'LotArea']
target = 'SalePrice'

# Ensure that the expected features exist in the dataset
features = [col for col in features if col in df.columns]
if not features:
    st.error("None of the specified features were found in the dataset.")
    st.stop()

X = df[features]
y = df[target]

if X.shape[0] < 2:
    st.error("Not enough data to split into train and test sets.")
    st.stop()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train the Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Streamlit Web App Interface
# -----------------------------
st.write("""
This app predicts the **Housing Price** based on user inputs.
Use the sidebar to adjust parameters and view the predicted sale price.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    input_data = {}
    for feature in features:
        min_val = int(X[feature].min())
        max_val = int(X[feature].max())
        mean_val = int(X[feature].mean())
        input_data[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

st.subheader('User Input Parameters')
st.write(input_df)

# Predict the housing price based on the input
prediction = model.predict(input_df)

st.subheader('Predicted Sale Price')
st.write(f"${prediction[0]:,.2f}")
