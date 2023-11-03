import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data_daily.csv'

# Load the dataset
df = pd.read_csv(file_path, parse_dates=['# Date'])

# Display the first few entries
print(df)

# Plot the daily receipt count
plt.figure(figsize=(15, 7))
plt.scatter(df['# Date'], df['Receipt_Count'], color='#FBA919', label='Daily Receipt Count', zorder=2)
plt.title('Daily Scanned Receipts over Time')
plt.xlabel('Date')
plt.ylabel('Number of Receipts')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0) 

plt.legend()

# Display the plot
plt.show()
