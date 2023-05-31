import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, concatenate, Bidirectional

# Load the data
data = pd.read_csv('cleaned_data.csv')  # Replace 'your_data.csv' with the path to your dataset

# Select the input features and output targets
input_features = ['in1', 'in4', 'in7', 'ex_attr_1', 'ex_attr_2', 'ex_attr_3', 'ex_attr_4', 'ex_attr_5']
output_targets = ['out1', 'out2', 'out3', 'out4', 'out5']

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=1392, random_state=42)

# Define the number of time steps and features
n_steps = 5  # Number of time steps
n_features = len(input_features)

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Normalize the input data
train_data_normalized = scaler.fit_transform(train_data[input_features])
test_data_normalized = scaler.transform(test_data[input_features])

y_train_data = train_data[output_targets]

X_train = []
y_train = []
X_test = []
y_test = []

# Generate training sequences
for i in range(n_steps, len(train_data_normalized)):
    X_train.append(train_data_normalized[i - n_steps:i, :n_features])
    y_train.append(y_train_data[i - n_steps:i])

# Generate testing sequences
for i in range(n_steps, len(test_data_normalized)):
    X_test.append(test_data_normalized[i - n_steps:i, :n_features])
    y_test.append(test_data_normalized[i, :len(output_targets)])

# Reshape training sequences
X_train = np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], n_steps, n_features))

y_train = np.array(y_train)
y_train = np.reshape(y_train, (y_train.shape[0], len(output_targets)))

# Reshape testing sequences
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], n_steps, n_features))

y_test = np.array(y_test)
y_test = np.reshape(y_test, (y_test.shape[0], len(output_targets)))

# Create the LSTM model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))
num_layers = 10
units = 128

previous_output = model.layers[-1].output

for _ in range(num_layers):
    current_output = LSTM(units, activation='relu', return_sequences=True)(previous_output)
    previous_output = concatenate([previous_output, current_output])

model.add(LSTM(units, activation='relu'))
model.add(Dense(len(output_targets)))

# Compile the model
model.compile(optimizer='adam', loss='mse')

model.fit(
    X_train, 
    train_data[output_targets].values, 
    epochs=1000, 
    batch_size=64, 
    validation_data=(
        X_test, 
        test_data[output_targets].values
    )
)

# Evaluate the model
loss = model.evaluate(X_test, test_data[output_targets].values)
print("Test Loss:", loss)


predict_data = pd.read_csv('predict_data.csv')
predict_data.drop(output_targets, inplace=True, axis=1)
predict_data_scaled = scaler.transform(predict_data[input_features].values)
X_pred = predict_data_scaled.reshape(-1, n_steps, n_features)

# Make predictions
predictions = model.predict(X_pred)

# Create a DataFrame with the predictions
prediction_data = pd.DataFrame(predictions, columns=output_targets)

# Add the predictions to the original data
output_data = pd.concat([predict_data, prediction_data], axis=1)

# Save the output data to a CSV file
output_data.to_csv('predictions.csv', index=False)
