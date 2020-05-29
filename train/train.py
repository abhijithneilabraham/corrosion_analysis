#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:10:58 2020

@author: abhijithneilabraham
"""

import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from math import sqrt
from numpy import random,concatenate,asarray

from sklearn.preprocessing import LabelEncoder
random.seed(7)

data =  pd.read_excel(r'datasheet.xlsx',sheet_name="data1")
y=list(data["OVERALL CONDITION"][1:])
x1=list(data["METAL LOSS"][1:])
x2=list(data["CATHODIC PROTECTION "][1:])

df=pd.DataFrame({"METAL LOSS":x1,"CATHODIC PROTECTION":x2,"OVERALL CONDITION":y})
values=df.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
x=scaled[:,:2]
y=scaled[:,2]

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,shuffle=False)
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
# compile the keras model

# fit the keras model on the dataset
print(train_x[0])
model.fit(train_x, train_y, epochs=150, batch_size=10)
model.save("final.h5")
from keras.models import load_model
model=load_model("final.h5")

yhat=model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
prediction = concatenate((test_x,yhat), axis=1)
prediction2 = scaler.inverse_transform(prediction)
pred = prediction2[:,-1]
test_y = test_y.reshape((len(test_y), 1))
actual_y = concatenate((test_y, test_x), axis=1)
actual_y2 = scaler.inverse_transform(actual_y)
act = actual_y2[:,-1]
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(act, pred))
print('Test RMSE: %.3f' % rmse)


pyplot.figure(1)
pyplot.plot(act, label='actual')
pyplot.plot(pred, label='predicted')
pyplot.legend()

pyplot.show()
# for i in range(len(yhat)):
#     print("actual==",inv_yhat[i],"  predicted",inv_test_y[i])


