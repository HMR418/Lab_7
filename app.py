import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
# Read the dataset from the local file in the GitHub repository
df = pd.read_excel('AmesHousing.xlsx')

# 2. Preprocess the data
# For simplicity, we drop rows with missing values.
# In practice, you might want to fill missing values or handle them differently.
data.dropna(inplace=True)

# Assuming that the target variable is 'Price' and the rest are features
X = data.drop('Price', axis=1)
y = data['Price']

# Optional: If there are categorical features, you would need to encode them.
# For example:
# X = pd.get_dummies(X)

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the feature data
# Scaling can help many models converge faster and achieve better performance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train a regression model (using Linear Regression as an example)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# 7. Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
