import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
bike_data = pd.read_csv('daily-bike-share.csv')

# Feature engineering
bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day

# Define features
numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']
label = bike_data['rentals']

# Show basic info
print(bike_data.head())
print(bike_data[numeric_features + ['rentals']].describe())  # numeric_features defined below




# Histograms & boxplot for label
fig, ax = plt.subplots(2, 1, figsize=(9,12))
ax[0].hist(label, bins=100)
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)
ax[0].set_ylabel('Frequency')
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Rentals')
fig.suptitle('Rental Distribution')
plt.show()

# Histograms for numeric features
for col in numeric_features:
    fig = plt.figure(figsize=(9,6))
    ax = fig.gca()
    feature = bike_data[col]
    feature.hist(bins=100, ax=ax)
    ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(col)
    plt.show()

# Bar plots for categorical features
for col in categorical_features:
    counts = bike_data[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9,6))
    ax = fig.gca()
    counts.plot.bar(ax=ax, color='steelblue')
    ax.set_title(col + ' counts')
    plt.show()

# Scatter plots with correlation
for col in numeric_features:
    correlation = bike_data[col].corr(label)
    plt.figure(figsize=(9,6))
    plt.scatter(bike_data[col], label)
    plt.xlabel(col)
    plt.ylabel('Bike Rentals')
    plt.title(f'rentals vs {col} - correlation: {correlation:.2f}')
    plt.show()

# Boxplots by categorical features
for col in categorical_features:
    fig = plt.figure(figsize=(9,6))
    ax = fig.gca()
    bike_data.boxplot(column='rentals', by=col, ax=ax)
    ax.set_title('Label by ' + col)
    plt.show()