import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import scipy.stats as stats
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_json('Assignment1/Dataminers 2024.json')
df['Your height (in International inches)'] = pd.to_numeric(df['Your height (in International inches)'], errors='coerce')

# Check the data types, missing values & basic statistics
df.info()

# drop rows with NaN values in the height and shoe size columns
df_BoxPlotHeight = df.dropna(subset=['Your height (in International inches)'])
df_BoxPlotShoeSize = df.dropna(subset=['Your mean shoe size (In the European Continental system)'])

# Box Plot for outlier detection for our numeric columns.
# Plot 1: Height
plt.subplot(1, 2, 1)
plt.boxplot(
    df_BoxPlotHeight['Your height (in International inches)'], vert=True, patch_artist=True,
    boxprops={
        "facecolor": "#2580B7"
    }
)
plt.title("Box Plot - Height")
plt.ylabel("Height (in International Inches)")
plt.ylim(0, 200)

# Plot 2: Shoe Size
plt.subplot(1, 2, 2)
plt.boxplot(
    df_BoxPlotShoeSize['Your mean shoe size (In the European Continental system)'], vert=True, patch_artist=True,
    boxprops={
        "facecolor": "#2580B7"
    }
)
plt.title("Box Plot - Shoe Size")
plt.ylabel("Shoe size (European Continental System)")
plt.ylim(0, 50)
plt.tight_layout()
plt.show()

# Function to convert height to cm
def convert_to_cm(height):
    if height < 10: 
        return height * 30.48
    elif height < 100: 
        return height * 2.54
    else:  
        return height

# Apply the convert_to_cm function to each value in the height column to new column in cm
df['Height in cm'] = df['Your height (in International inches)'].apply(convert_to_cm)

# Converts shoe sizes to numeric and replaces non-numeric or values under 20 and over 60 with NaN to remove misstyped values
df['Your mean shoe size (In the European Continental system)'] = pd.to_numeric(df['Your mean shoe size (In the European Continental system)'], errors='coerce')
df.loc[df['Your mean shoe size (In the European Continental system)'] < 20, 'Your mean shoe size (In the European Continental system)'] = np.nan
df.loc[df['Your mean shoe size (In the European Continental system)'] > 60, 'Your mean shoe size (In the European Continental system)'] = np.nan

numberData = ['Height in cm', 'Your mean shoe size (In the European Continental system)'] # Left out How many letters are there in the word \"Seattle\"? as we will not be using it in the analysis

# Convert non-numeric values to NaN
df_numeric = df[numberData].apply(pd.to_numeric, errors='coerce')

# Initialize KNN and impute missing values
knn = KNNImputer(n_neighbors=2)
df[numberData] = knn.fit_transform(df_numeric)

# Calculate Q1 (25th percentile) and Q3 (75th percentile) of height to calculate IQR
Q1 = df['Height in cm'].quantile(0.25)
Q3 = df['Height in cm'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers and remove them
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Height in cm'] >= lower_bound) & (df['Height in cm'] <= upper_bound)]

# Pearson correlation coefficient
pearson_corr = pearsonr(df['Your mean shoe size (In the European Continental system)'], df['Height in cm'])
print(f"Pearson correlation coefficient: {pearson_corr[0]}")

### Training a simple linear regression model
X = df[['Your mean shoe size (In the European Continental system)']]  # Predictor (shoe size)
y = df['Height in cm']  # Target (height)

# Doing a 80% train, 20% test split on the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_predL = model.predict(X_test)

print(f"Model Coefficient (Slope): {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")

# Calculate error metrics for linear model
linear_r2 = r2_score(y_test, y_predL)
linear_mae = mean_absolute_error(y_test, y_predL)
linear_mse = mean_squared_error(y_test, y_predL)
print(f"R-squared for linear model: {linear_r2}")
print(f"MAE for linear model: {linear_mae}")
print(f"MSE for linear model: {linear_mse}")

### Training a polynomial regression model
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
model.fit(X_train_poly, y_train)
y_predP = model.predict(X_test_poly)

# Getting feature names (useful to know which term the coefficients correspond to)
feature_names = poly.get_feature_names_out()

# Calculate error metrics for polynomial model
poly_r2 = r2_score(y_test, y_predP)
poly_mae = mean_absolute_error(y_test, y_predP)
poly_mse = mean_squared_error(y_test, y_predP)
print(f"R-squared for polynomial model: {poly_r2}")
print(f"MAE for polynomial model: {poly_mae}")
print(f"MSE for polynomial model: {poly_mse}")

# Plotting the pearson correlation
plt.subplot(2, 2, 1)
plt.scatter(X, y, color='grey', label='Cleaned data')
plt.xlabel('Shoe size')
plt.ylabel('Height (cm)')
plt.title("Pearson correlation (r2 = 0.8889)")
plt.legend()

# Plotting both the linear and polynomial models on cleaned data
plt.subplot(2, 2, 2)
plt.scatter(X, y, color='grey', label='Cleaned data')
plt.plot(X_test, y_predL, color='red', label='Linear model')
plt.plot(X_test, y_predP, color='blue', label='Polynomial model')
plt.xlabel('Shoe size')
plt.ylabel('Height (cm)')
plt.title("Linear and Polynomial Regression")
plt.legend()

# Plotting the linear regression model on test data
plt.subplot(2, 2, 3)
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_test, y_predL, color='blue', linewidth=3, label='Linear model')
plt.xlabel('Shoe size')
plt.ylabel('Height (cm)')
plt.title("Linear Regression (r2 = 0.7875)")
plt.legend()

# Plotting the polynomial model on test data
plt.subplot(2, 2, 4)
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_test, y_predP, color='red', label='Polynomial model')
plt.xlabel('Shoe size')
plt.ylabel('Height (cm)')
plt.title("Polynomial Regression (r2 = 0.7984)")
plt.legend()
plt.tight_layout()
plt.show()