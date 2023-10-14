import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("cleaned_data.csv")
data = df.loc[df['Entity'] == 'Greece'].copy() #Locate all the data for Greece
data['Daily Cases'] = 0     #initialize Daily Cases with zeros

for i in range(1, len(data)):   #For all the data for Greece
    data.iloc[i, 15] = data.iloc[i, 13] - data.iloc[i - 1, 13]  #Find the cases of every day and save them to Daily Cases column
data['Positivity'] = (data['Daily Cases'] / data['Daily tests'])    # Find the positivity of every day
data.replace([np.inf, -np.inf], 0, inplace=True)    #Replace inf values with zero
data.replace(np.nan, 0, inplace=True)   #Replace nan values with zero
data = data.iloc[:, [11, 12, 13, 16]]
data.to_csv('Greece.csv', index=False, encoding="utf-8-sig") #Save the new dataframe that contains Date, Daily tests, Cases, Positivity for Greece


df = pd.read_csv('Greece.csv')
x = df.index[df['Date'] == '2021-01-01'].tolist()   #Find the index where Date == 2021-01-01

data2 = df.filter(['Cases']).copy()
dataset = data2.values


scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)# Scale data for better results


x_train = []
y_train = []


for i in range(x[0]):     #For the days before 2021-01-01
    x_train.append(scaled_data[i])   #X_training set equals i
    y_train.append(scaled_data[i+3])    #Y_train set equals i+3

x_train, y_train = np.array(x_train), np.array(y_train)

x_test = []
y_test = []

for i in range(x[0], len(scaled_data)-3):     #For the days after 2021-01-01
    x_test.append(scaled_data[i])    #X_test set equals i
    y_test.append(scaled_data[i+3])      #Y_test set equals i+6

x_test = np.array(x_test)
y_test = scaler.inverse_transform(y_test)   #invert scaling for Y_test

regressor = SVR(kernel='linear', gamma='auto', C=2).fit(x_train, y_train.ravel())   #Initialize
predictions = regressor.predict(x_test)     #Make prediction for X_test
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))  #Invert scaling for the predicted data


train = df[:x[0]].copy()  # Create tables for plotting
valid = df[x[0]+3:].copy()
valid['Predictions'] = predictions  #Insert predictions to the dataframe
print(train)
print(valid)

df['Prediction_Daily_Cases'] = 0
df['Prediction_Positivity'] = np.nan

for i in range(1, len(valid['Predictions'])):
    df.iloc[x[0] + i, 4] = (predictions[i] - predictions[i - 1])    #Find the Cases prediction for every day
df.iloc[x[0]:, 5] = df['Prediction_Daily_Cases'].iloc[x[0]:] / df['Daily tests'].iloc[x[0]:]    #Calculate the predicted pisitivity
df.replace([np.inf, -np.inf], 0, inplace=True)
df.replace(np.nan, 0, inplace=True)

plt.figure(figsize=(16, 8))  # Plot results
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Power demand')
plt.plot(train['Cases'])
plt.plot(valid['Cases'], color='red', alpha=0.5)
plt.plot(valid['Predictions'], color='blue', alpha=0.5)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

plt.figure(figsize=(16, 8))  # Plot results
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Power demand')
plt.plot(df['Positivity'].iloc[:x[0]])
plt.plot(df['Positivity'].iloc[x[0]:], color='red', alpha=0.5)
plt.plot(df['Prediction_Positivity'].iloc[x[0]:], color='blue', alpha=0.5)
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

mae = mean_absolute_error(valid['Cases'], valid['Predictions']) #Mean absolute error
print('mae:' + str(mae))
mse = mean_squared_error(valid['Cases'], valid['Predictions'])  #Mean Squared error
print('mse :' + str(mse))
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)  #RMSE
print('rmse :' + str(rmse))
r2 = r2_score(valid['Cases'], valid['Predictions']) #R2 score
print('r2:' + str(r2))

df.to_csv('Greece_data.csv', index=False, encoding="utf-8-sig") #Save data for Greece

