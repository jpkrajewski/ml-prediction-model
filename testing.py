import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load machine data
data = pd.read_excel('datafolder/input_data_manufacturing_process.xls')

# Select the relevant portion of the data
data_with_output = data.iloc[:7391]

# Delete outliers
z_scores = np.abs((data_with_output - data_with_output.mean()) / data_with_output.std())
threshold = 5
outlier_mask = z_scores < threshold
data_no_outliers = data_with_output.copy()
data_no_outliers[~outlier_mask] = np.nan
data_no_outliers = data_no_outliers.interpolate()

# Extract relevant columns
input_columns = ['in1', 'in2', 'in3', 'in4', 'in6', 'in7', 'in8', 'in24', 'in29']
output_columns = ['out1', 'out2', 'out3', 'out4', 'out5']
important_inputs = data_no_outliers[input_columns + output_columns]

# Calculate mean of highly correlated inputs
correlated_inputs_group_1 = ['in22', 'in23', 'in25', 'in26']
correlated_inputs_group_2 = ['in21', 'in27']
correlated_inputs_group_3 = ['in20', 'in28', 'in21', 'in27']
correlated_inputs_group_4 = ['in11', 'in12', 'in13', 'in14', 'in15', 'in16', 'in17', 'in18', 'in19']
correlated_inputs_group_5 = ['in5', 'in9', 'in10']
important_inputs['ex_attr_1'] = data_no_outliers[correlated_inputs_group_1].mean(axis=1)
important_inputs['ex_attr_2'] = data_no_outliers[correlated_inputs_group_2].mean(axis=1)
important_inputs['ex_attr_3'] = data_no_outliers[correlated_inputs_group_3].mean(axis=1)
important_inputs['ex_attr_4'] = data_no_outliers[correlated_inputs_group_4].mean(axis=1)
important_inputs['ex_attr_5'] = data_no_outliers[correlated_inputs_group_5].mean(axis=1)

# Drop unnecessary columns
less_inputs = important_inputs.drop(['in2', 'in29', 'in8', 'in24', 'in3', 'in6'], axis=1)

# Save the cleaned data
less_inputs.to_csv('cleaned_data.csv', index=False)

# Print columns for verification
print(less_inputs.columns)