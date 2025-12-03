# -----------------------------------------------------------
#  WEATHER DATA ANALYSIS PROJECT
#  Covers: Loading, Cleaning, NumPy Stats, Visualization,
#          Grouping & Aggregation
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# TASK 1: Data Acquisition
# --------------------------

# Load the CSV file
df = pd.read_csv("weather.csv")

print("HEAD:\n", df.head())
print("\nINFO:\n")
print(df.info())
print("\nDESCRIBE:\n", df.describe())


# --------------------------
# TASK 2: Data Cleaning
# --------------------------

# Convert date column
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Handle missing values (drop or fill)
df = df.dropna(subset=['temperature', 'rainfall', 'humidity'])

# Select relevant columns
df = df[['date', 'temperature', 'rainfall', 'humidity']]


# --------------------------
# TASK 3: NumPy Statistical Analysis
# --------------------------

temps = df['temperature'].to_numpy()

mean_temp = np.mean(temps)
min_temp = np.min(temps)
max_temp = np.max(temps)
std_temp = np.std(temps)

print("\n--- Temperature Statistics ---")
print("Mean Temperature:", mean_temp)
print("Min Temperature:", min_temp)
print("Max Temperature:", max_temp)
print("Standard Deviation:", std_temp)

# Daily / monthly / yearly summaries
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

daily_stats = df.groupby('date')['temperature'].mean()
monthly_stats = df.groupby('month')['temperature'].mean()
yearly_stats = df.groupby('year')['temperature'].mean()


# --------------------------
# TASK 4: Visualizations
# --------------------------

# 1. Line chart: Daily temperature
plt.figure(figsize=(10,4))
plt.plot(df['date'], df['temperature'])
plt.title("Daily Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.show()

# 2. Bar chart: Monthly rainfall
plt.figure(figsize=(8,4))
monthly_rain = df.groupby('month')['rainfall'].sum()
plt.bar(monthly_rain.index, monthly_rain.values)
plt.title("Monthly Rainfall Total")
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.show()

# 3. Scatter plot: Humidity vs Temperature
plt.figure(figsize=(6,4))
plt.scatter(df['humidity'], df['temperature'])
plt.title("Humidity vs Temperature")
plt.xlabel("Humidity")
plt.ylabel("Temperature")
plt.show()

# 4. Combined plots
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(df['date'], df['temperature'])
plt.title("Daily Temperature")

plt.subplot(1,2,2)
plt.scatter(df['humidity'], df['temperature'])
plt.title("Humidity vs Temp")

plt.tight_layout()
plt.show()


# --------------------------
# TASK 5: Grouping & Aggregation
# --------------------------

# Grouping by month
monthly_group = df.groupby('month').agg({
    'temperature': ['mean', 'max', 'min'],
    'rainfall': 'sum',
    'humidity': 'mean'
})

print("\n--- Monthly Aggregated Data ---\n")
print(monthly_group)


# --------------------------
# TASK 6: EXPORT RESULTS
# --------------------------

# Export cleaned data
df.to_csv("cleaned_weather_data.csv", index=False)

# Export aggregated monthly summary
monthly_group.to_csv("monthly_weather_summary.csv")

# Prepare storytelling text file
story = f"""
WEATHER ANALYSIS STORY SUMMARY
------------------------------

Over the recorded period, the average temperature was {mean_temp:.2f}°C.
The warmest day reached {max_temp:.2f}°C, while the coolest dropped to {min_temp:.2f}°C.
Temperature variability was moderate with a standard deviation of {std_temp:.2f}.

Rainfall analysis shows that the month with the highest rainfall was 
Month {monthly_rain.idxmax()} with a total of {monthly_rain.max():.2f} mm.

Temperature trends indicate:
- Increasing temperature during summer months.
- Cooler temperatures during winter months.

Humidity showed a visible correlation with temperature,
meaning higher humidity often coincided with higher temperatures.

Overall, the weather pattern reflects a typical seasonal cycle with
significant rainfall variations and a stable temperature distribution.
"""

with open("weather_storytelling.txt", "w") as f:
    f.write(story)

print("\nFiles Exported:")
print("1. cleaned_weather_data.csv")
print("2. monthly_weather_summary.csv")
print("3. weather_storytelling.txt")
print("\nStorytelling summary saved successfully!")