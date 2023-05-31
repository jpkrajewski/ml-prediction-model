import pandas
import numpy as np
import datashower

# Load machine data
data = pandas.read_excel('datafolder/input_data_manufacturing_process.xls')
# Data with input-out puts only to 7391 row, select only that
data_with_output = data.iloc[:7391]
timestep_labels = data_with_output['timestep']

"""Delete outliers"""

# Calculate z-scores
z_scores = np.abs((data - data.mean()) / data.std())
# Define threshold for outlier detection (e.g., z-score > 3)
threshold = 3
# Create a mask to identify outliers
mask = z_scores < threshold
# Filter out outliers
data_no_outliers = data_with_output.copy()
data_no_outliers[~mask] = np.nan  # Set outliers to NaN
data_no_outliers = data_no_outliers.interpolate() # Interpolate missing values


"""Dimensionality Reduction

1. Find highly corelated attributes
2. Extract attributes

"""

# 'in22', 'in23', 'in25', 'in26'
# highly_correlated_inputs_group_1 = data_no_outliers.columns[data_no_outliers.min() > -450]

# 'in21', 'in27'
mask = (data_no_outliers >= -550) & (data_no_outliers <= -500)
highly_correlated_inputs_group_2 = data_no_outliers.columns[mask.all()]

# 'in20', 'in28', 'in21', 'in27'
mask = (data_no_outliers >= -570) & (data_no_outliers <= -502)
highly_correlated_inputs_group_3 = data_no_outliers.columns[mask.all()]

# 'in11', 'in12', 'in13', 'in14', 'in15', 'in16', 'in17', 'in18', 'in19'
mask = (data_no_outliers >= -660) & (data_no_outliers <= -640)
highly_correlated_inputs_group_4 = data_no_outliers.columns[mask.all()]

# 'in5', 'in9', 'in10'
mask = (data_no_outliers >= -640) & (data_no_outliers <= -630)
highly_correlated_inputs_group_5 = data_no_outliers.columns[mask.all()]

# 'in1', 'in2', 'in3', 'in4', 'in6', 'in7', 'in8', 'in24', 'in29'
mask = (data_no_outliers >= -640) & (data_no_outliers <= -604)
inputs_group_6 = data_no_outliers.columns[mask.all()]

# extracted = pandas.DataFrame()
# extracted['timestep'] = timestep_labels
# extracted['ex_atrr_1'] = data_no_outliers[highly_correlated_inputs_group_1].mean(axis=1)
# extracted['ex_atrr_2'] = data_no_outliers[highly_correlated_inputs_group_2].mean(axis=1)
# extracted['ex_atrr_3'] = data_no_outliers[highly_correlated_inputs_group_3].mean(axis=1)
# extracted['ex_atrr_4'] = data_no_outliers[highly_correlated_inputs_group_4].mean(axis=1)
# extracted['ex_atrr_5'] = data_no_outliers[highly_correlated_inputs_group_5].mean(axis=1)

important_inputs = pandas.DataFrame()
important_inputs['timestep'] = timestep_labels
important_inputs[inputs_group_6] = data_no_outliers[inputs_group_6]

# extracted = pandas.concat([extracted, important_inputs], axis=1)

print(important_inputs)
# data = np.append(data_no_outliers[inputs_group_6], data_with_output[['out1', 'out2', 'out3', 'out4', 'out5']])
# normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

less_inputs = important_inputs.drop(['in2', 'in10', 'in9', 'in29', 'in8', 'in5', 'in24', 'in3', 'in6'], axis=1)
less_inputs['ex_atrr_1'] = data_no_outliers[['in22', 'in23', 'in25', 'in26']].mean(axis=1)
less_inputs['ex_atrr_2'] = data_no_outliers[highly_correlated_inputs_group_2].mean(axis=1)
less_inputs['ex_atrr_3'] = data_no_outliers[highly_correlated_inputs_group_3].mean(axis=1)
less_inputs['ex_atrr_4'] = data_no_outliers[highly_correlated_inputs_group_4].mean(axis=1)
less_inputs['ex_atrr_5'] = data_no_outliers[highly_correlated_inputs_group_5].mean(axis=1) 

less_inputs['out1'] = data_no_outliers['out1']
less_inputs['out2'] = data_no_outliers['out2']
less_inputs['out3'] = data_no_outliers['out3']
less_inputs['out4'] = data_no_outliers['out4']
less_inputs['out5'] = data_no_outliers['out5']

print(less_inputs.columns)

datashower.show(less_inputs)
print(data.isnull().sum())
less_inputs.to_csv('cleaned_data.csv')