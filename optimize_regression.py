import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

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

# === 1. Simple Gradient Boosting ===
model = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
predictions = model.predict(X_test)

print("GradientBoosting:")
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2:", r2_score(y_test, predictions))

# === 2. Grid Search (best parameters) ===
from sklearn.model_selection import GridSearchCV

params = {
    'learning_rate': [0.1, 0.5, 1.0],
    'n_estimators': [50, 100, 150]
}

gridsearch = GridSearchCV(GradientBoostingRegressor(random_state=0), 
                          params, scoring='r2', cv=3)
gridsearch.fit(X_train, y_train)

print("\nBest params:", gridsearch.best_params_)
print("Best R2:", gridsearch.best_score_)

model = gridsearch.best_estimator_
predictions = model.predict(X_test)
print("\nTuned GradientBoosting:")
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2:", r2_score(y_test, predictions))

# === 3. Pipeline with preprocessing ===
numeric_features = [6, 7, 8, 9]      # indices of temp, atemp, hum, windspeed
categorical_features = [0, 1, 2, 3, 4, 5]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=0))
])

model = pipeline.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nRandomForest with Pipeline:")
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2:", r2_score(y_test, predictions))

# Save model
joblib.dump(model, './bike-share.pkl')
print("\nModel saved as bike-share.pkl")

# === Prediction example ===
X_new = np.array([[1,1,0,3,1,1,0.226957,0.22927,0.436957,0.1869]])
loaded_model = joblib.load('./bike-share.pkl')
result = loaded_model.predict(X_new)
print(f'\nPrediction for new sample: {np.round(result[0]):.0f} rentals')



# An array of features based on five-day weather forecast
X_new = np.array([[0,1,1,0,0,1,0.344167,0.363625,0.805833,0.160446],
                  [0,1,0,1,0,1,0.363478,0.353739,0.696087,0.248539],
                  [0,1,0,2,0,1,0.196364,0.189405,0.437273,0.248309],
                  [0,1,0,3,0,1,0.2,0.212122,0.590435,0.160296],
                  [0,1,0,4,0,1,0.226957,0.22927,0.436957,0.1869]])




# Use the model to predict rentals
results = loaded_model.predict(X_new)
print('5-day rental predictions:')
for prediction in results:
    print(np.round(prediction))