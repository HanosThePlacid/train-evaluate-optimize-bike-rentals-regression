import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
bike_data = pd.read_csv('daily-bike-share.csv')
bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day

# Features and target
X = bike_data[['season','mnth','holiday','weekday','workingday','weathersit',
               'temp','atemp','hum','windspeed']].values
y = bike_data['rentals'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print(f'Training Set: {X_train.shape[0]} rows')
print(f'Test Set: {X_test.shape[0]} rows\n')

# 1. Lasso Regression
from sklearn.linear_model import Lasso
model = Lasso().fit(X_train, y_train)
predictions = model.predict(X_test)

print("Lasso Regression:")
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2:", r2_score(y_test, predictions))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Lasso Regression Predictions')
plt.plot(y_test, np.poly1d(np.polyfit(y_test, predictions, 1))(y_test), color='magenta')
plt.show()

# 2. Decision Tree
from sklearn.tree import DecisionTreeRegressor, export_text
model = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
print("\nDecision Tree:")
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))
print("RMSE:", np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
print("R2:", r2_score(y_test, model.predict(X_test)))

tree = export_text(model, max_depth=3)
print("\nDecision Tree (first 3 levels):\n", tree)

# 3. Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nRandom Forest:")
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2:", r2_score(y_test, predictions))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Random Forest Predictions')
plt.plot(y_test, np.poly1d(np.polyfit(y_test, predictions, 1))(y_test), color='magenta')
plt.show()

# 4. Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nGradient Boosting:")
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2:", r2_score(y_test, predictions))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Gradient Boosting Predictions')
plt.plot(y_test, np.poly1d(np.polyfit(y_test, predictions, 1))(y_test), color='magenta')
plt.show()

print("\nAll models trained and evaluated.")