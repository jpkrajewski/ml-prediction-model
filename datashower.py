import pandas as pd
import matplotlib.pyplot as plt

real_data = pd.read_csv('cleaned_data.csv')
prediction_data = pd.read_csv('predictions_2.csv')

df = pd.concat([real_data, prediction_data], ignore_index=True)

timestep = range(len(df))
_, ax = plt.subplots(figsize=(10, 6))

df = df[['out1', 'out2', 'out3', 'out4', 'out5']]

for column in df.columns:
    ax.plot(timestep, df[column], label=column)

# Set title, labels, and legend
ax.set_title('Input Values Over Time')
ax.set_xlabel('Time Step')
ax.set_ylabel('Value')
ax.legend()

# Show the plot
plt.show()