#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data Collection and Analysis
parkinsons_data = pd.read_csv('parkinsons_data.csv')
print(parkinsons_data.head())
print(parkinsons_data.shape)
print(parkinsons_data.info())
print(parkinsons_data.isnull().sum())

#getting some statistical measures about the data
print(parkinsons_data.describe())

#distribution of target variable
#1---> parkinsons positive
#0---> healthy
print(parkinsons_data['status'].value_counts())

#grouping the data based on their target value
print(parkinsons_data.groupby('status').mean(numeric_only=True))

#Data Pre-Processing, seprating the features and target
x = parkinsons_data.drop(columns=['name', 'status'], axis=1)
y = parkinsons_data['status']
print(x)
print(y)

#Spliting the data to training data & test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

#Data Standardization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train)

#Model Training, support vector machine model(svm)
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

#Model Evaluation, Accuracy Score
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy Score of training data: ', training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy Score of test data: ', test_data_accuracy)

#Building a predictive system
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
#standarize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have Parkinsons Disease')
else:
    print('The Person has Parkinsons')