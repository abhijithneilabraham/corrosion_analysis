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

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
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
encoder = LabelEncoder()
train_y=encoder.fit_transform(y)
train_x,test_x,train_y,test_y=train_test_split(x,train_y,test_size=0.3)
# model = Sequential()
# model.add(Dense(12, input_dim=2, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='softmax'))
# # compile the keras model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit the keras model on the dataset
# model.fit(train_x, train_y, epochs=150, batch_size=20)
# model.save("final.h5")
from keras.models import load_model
model=load_model("final.h5")
_, accuracy = model.evaluate(test_x, test_y)
print('Accuracy: %.2f' % (accuracy*100))
yhat=model.predict_classes(test_x)

inv_yhat = encoder.inverse_transform(yhat)
inv_yhat = concatenate(( inv_yhat[:,None],test_x),axis=1)
inv_yhat2=scaler.inverse_transform(inv_yhat)
print(values[-10:,2],inv_yhat2[-10:,2])
pyplot.figure(1)
pyplot.plot(values[-50:,2], label='actual')
pyplot.plot(inv_yhat2[-50:,2], label='predicted')
pyplot.legend()

pyplot.show()
# for i in range(len(yhat)):
#     print("actual==",inv_yhat[i],"  predicted",inv_test_y[i])


