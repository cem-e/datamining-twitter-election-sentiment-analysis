#importing relevant libraries/packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, linregress


# LOADING AND PREPROCESSING THE DATA 

### LOADING DATA
df = pd.read_csv('Assignment2(exam_assignment)/Depression_Professional_Dataset.csv')

### PREPROCESSING
print("Total missing values:\n", df.isna().sum().sum(), "\n\n")
print("Missing values in the different features:\n", df.isna().sum())


# DATA EXPLORATION & VISUALIZATION

print(df.head())

print(df.info())

### EXPLORING COLUMNS OF INTEREST

df['Depression'].value_counts().plot(kind='bar', title='Depression')
plt.xlabel('Response')
plt.ylabel('Count')
plt.show()

df['Job Satisfaction'].plot(kind='hist', bins=20, title='Job Satisfaction Distribution')
plt.xlabel('Job Satisfaction')
plt.ylabel('Frequency')
plt.show()

df['Dietary Habits'].value_counts().plot(kind='bar', title='Dietary Habits')
plt.xlabel('Response')
plt.ylabel('Count')
plt.show()

sns.heatmap(df.corr(numeric_only=True))





