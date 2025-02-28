import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------

# Read the dataset from the local file in the GitHub repository
df = pd.read_excel('AmesHousing.xlsx')

# For simplicity, drop rows with missing values
df = df.dropna()

# Define the features and the target variable
# (Ensure that these columns exist in your dataset)
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'LotArea']
target = 'SalePrice'

# If needed, adjust the features list to match the available columns in your dataset
features = [col for col in features if col in df.columns]

# Split the dataset into feature matrix X and target vector y
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model (using Linear Regression here)
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Streamlit Web App
# -----------------------------

st.title('Housing Price Prediction App')
st.write("""
This app predicts the **Housing Price** based on user input parameters.
Adjust the values in the sidebar and see the predicted sale price.
""")

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    input_data = {}
    # Create a slider for each feature using its min, max, and mean values
    for feature in features:
        min_val = int(X[feature].min())
        max_val = int(X[feature].max())
        mean_val = int(X[feature].mean())
        input_data[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)
    # Convert the user input to a DataFrame
    input_df = pd.DataFrame(input_data, index=[0])
    return input_df

input_df = user_input_features()

# Display the user input parameters
st.subheader('User Input Parameters')
st.write(input_df)

# Predict housing price using the trained model
prediction = model.predict(input_df)

# Display the prediction
st.subheader('Predicted Sale Price')
st.write(f"${prediction[0]:,.2f}")
