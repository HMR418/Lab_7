import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_data():
# Read the dataset from the local file in the GitHub repository
    df = pd.read_excel('AmesHousing.xlsx')
    return data
except Exception as e:
    st.error(f"Error loading data: {e}")
        return None

def main():
    st.title("Housing Price Prediction")
    
    # Load data
    data = load_data()
    if data is None:
        return  # Stop if data loading fails

    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Preprocess the data
    data.dropna(inplace=True)  # Remove missing values
    if 'Price' not in data.columns:
        st.error("The dataset does not contain a 'Price' column.")
        return
    
    X = data.drop('Price', axis=1)
    y = data['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a regression model (Linear Regression)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write("Mean Squared Error:", mse)
    st.write("R2 Score:", r2)

if __name__ == "__main__":
    main()