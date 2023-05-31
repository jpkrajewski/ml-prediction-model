import pandas
import numpy
import sklearn

# Load machine data
data = pandas.read_excel('input_data_manufacturing_process.xls')

# Data with input-out puts only to 7391 row, select only that
data_with_output = data.iloc[:7391]

# Returns False, no data is missing
print(data_with_output.isnull().values.any())

timestep_column = data_with_output['timestep']
outputs_inputs = data_with_output.loc[:, data_with_output.columns != 'timestep']

# Normalizing data
normalized_outputs_inputs = (outputs_inputs - outputs_inputs.min()) / (outputs_inputs.max() - outputs_inputs.min())

# Concactenate
normalized_df = pandas.concat([timestep_column, normalized_outputs_inputs], axis=1)
print(normalized_df)

# Save normalized file
normalized_df.to_excel('normalized_data.xlsx', index=False)