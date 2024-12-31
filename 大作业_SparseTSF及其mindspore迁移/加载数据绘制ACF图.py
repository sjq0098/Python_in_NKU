import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Load the ETTh2 dataset (adjust the path if needed)
file_path = "path_to_your_ETTh2.csv"  # Adjust to your actual file path
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the data
print(df.head())

# Assuming the 'value' column contains the time-series data of interest (replace with actual column name)
# You may need to adjust this based on the dataset's structure (e.g., 'value', 'temperature', 'traffic', etc.)
time_series = df['value'].values  # Replace 'value' with the correct column name

# If there is a timestamp column, you can set it as the index (optional)
# df['timestamp'] = pd.to_datetime(df['timestamp'])  # Uncomment if you have a timestamp column
# df.set_index('timestamp', inplace=True)  # Uncomment if you set timestamp as index

# Check for missing values and handle them (optional)
if np.any(np.isnan(time_series)):
    print("Missing values detected! Interpolating...")
    time_series = pd.Series(time_series).interpolate().values

# Plot the ACF
plt.figure(figsize=(10, 6))
plot_acf(time_series, lags=80, ax=plt.gca(), alpha=0.05)

# Set plot title and labels
plt.title('Autocorrelation Function (ACF) of ETTh2 Dataset')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.grid(True)

# Show the plot
plt.show()
