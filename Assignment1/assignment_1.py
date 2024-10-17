
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Load the data
df = pd.read_json('Datamining2024/Assignment1/Dataminers 2024.json')
df['Your height (in International inches)'] = pd.to_numeric(df['Your height (in International inches)'], errors='coerce')

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

# Initialize KNN
knn = KNNImputer(n_neighbors=2)

# Perform KNN imputation
df[numberData] = pd.DataFrame(knn.fit_transform(df_numeric), columns=df_numeric.columns) # consider changing

# Calculate Q1 (25th percentile) and Q3 (75th percentile) of height
Q1 = df['Height in cm'].quantile(0.25)
Q3 = df['Height in cm'].quantile(0.75)

# Calculate IQR (Interquartile Range)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find rows where the 'Height in cm' column contains outliers
outliers = df[(df['Height in cm'] < lower_bound) | (df['Height in cm'] > upper_bound)]

# Remove outliers
df = df[(df['Height in cm'] >= lower_bound) & (df['Height in cm'] <= upper_bound)]

# Pearson correlation coefficient
pearson_corr = pearsonr(df['Your mean shoe size (In the European Continental system)'], df['Height in cm'])
print(f"Pearson correlation coefficient: {pearson_corr[0]}")

# Training a simple linear regression model
from sklearn.linear_model import LinearRegression

X = df[['Your mean shoe size (In the European Continental system)']]  # Predictor (shoe size)
y = df['Height in cm']  # Target (height)

# Doing a 80% train, 20% test split on the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Model Coefficient (Slope): {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")

# Define x (shoe size) and y (height)
x = df['Your mean shoe size (In the European Continental system)']
y = df['Height in cm']
