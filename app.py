import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    st.title("Housing Price Prediction App")
    st.write("This app predicts housing prices using the Ames Housing dataset.")

    # Read the dataset from the local file in the GitHub repository
    try:
        df = pd.read_excel('AmesHousing.xlsx')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Data Preprocessing: Drop missing values (or consider imputing missing values)
    df_clean = df.dropna()
    if df_clean.empty:
        st.error("The dataset is empty after dropping missing values. Consider using a different strategy (e.g., imputing missing values) or check the data file.")
        return
    st.write("Data after dropping missing values:")
    st.write(df_clean.head())

    # Ensure the target column exists
    if 'SalePrice' not in df_clean.columns:
        st.error("The dataset must contain a 'SalePrice' column.")
        return

    # Define features and target
    X = df_clean.drop('SalePrice', axis=1)
    y = df_clean['SalePrice']

    # Convert categorical columns to numeric using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    if X.empty:
        st.error("No features available after preprocessing. Please check the preprocessing steps.")
        return
    st.write("Features after encoding:")
    st.write(X.head())

    # Train-Test Split
    st.header("Train-Test Split")
    test_size = st.slider("Select test set fraction", 0.1, 0.5, 0.2, step=0.05)
    
    # Check if there is enough data to perform a split
    if len(X) < 2:
        st.error("Not enough data to split into training and testing sets.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    st.write(f"Training samples: {X_train.shape[0]}")
    st.write(f"Testing samples: {X_test.shape[0]}")

    # Model Training
    st.header("Model Training")
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.write("Model training complete.")

    # Model Evaluation
    st.header("Model Evaluation")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R² Score:", r2)

    # Display a sample of actual vs. predicted values
    st.subheader("Actual vs. Predicted Prices")
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.write(results.head())

if __name__ == '__main__':
    main()
