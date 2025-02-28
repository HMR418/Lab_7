import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.title("Housing Price Prediction App")
    st.write("This app predicts housing prices using the Ames Housing dataset.")

    # Read the dataset from the local file in the GitHub repository
    df = pd.read_excel('AmesHousing.xlsx')
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Data Preprocessing
    st.header("Data Preprocessing")
    # For simplicity, drop rows with missing values
    df = df.dropna()
    st.write("Data after dropping missing values:")
    st.write(df.head())

    # Ensure the target column exists
    if 'SalePrice' not in df.columns:
        st.error("The dataset must contain a 'SalePrice' column.")
        return

    # Define features and target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # Convert categorical columns to numeric using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    st.write("Features after encoding:")
    st.write(X.head())

    # Split data into training and testing sets
    st.header("Train-Test Split")
    test_size = st.slider("Select test set fraction", 0.1, 0.5, 0.2, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    st.write(f"Training samples: {X_train.shape[0]}")
    st.write(f"Testing samples: {X_test.shape[0]}")

    # Train a regression model (Linear Regression)
    st.header("Model Training")
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.write("Model training complete.")

    # Make predictions and evaluate the model
    st.header("Model Evaluation")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R² Score:", r2)

    # Display a sample of actual vs predicted values
    st.subheader("Actual vs. Predicted Prices")
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.write(results.head())

if __name__ == '__main__':
    main()
