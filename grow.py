'''
Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import math;
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential, Model
from keras.layers.core import Dense,Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
import matplotlib.animation as animation
from keras.layers import Input, Embedding, Merge
import csv
import keras

def gen_data(instream, batch_size):

    raw_data = []
    for row in instream:
        raw_data.append(((math.log(float(row[0])), float(row[4]), float(row[5])),))

    data_size = len(raw_data) - len(raw_data) % batch_size

    data = raw_data[0:data_size]
    
    return np.array(data)

def exp_result(data):
    
    for row in data:
        row[0] = math.exp(row[0])

    return data

    
def gen_datas(paths, batch_size):
    datas = []

    for path in paths:
        instream = csv.reader(open(path, 'r'))
        datas.append(gen_data(instream, batch_size))

    return datas

def sum_series_data(data):
    lastrow = np.array(((0,0,0),))
    summed_data = []

    for row in data:
        lastrow = np.add(lastrow,row)
        summed_data.append(lastrow)

    return np.array(summed_data)

def sum_result_data(data):
    lastrow = np.array((0,0,0))
    summed = []

    for row in data:
        lastrow = np.add(row,lastrow)
        summed.append(list(lastrow))
    
    return np.array(summed)

def create_expected(data):
    expected = []
    
    for row in data:
        expected.append(row[0])

    expected = np.array(expected)

    return np.append(expected[1:],expected[0:1], axis=0)

# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 100
epochs = 5
inner = 50
learning_rate = .001


print('Generating Data')
data1, data2, data3 = gen_datas(('datasets/set1.csv', 'datasets/set2.csv', 'datasets/set3.csv'), batch_size)


summed_data = sum_series_data(data2)

fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot(summed_data[:,0][:,1], summed_data[:,0][:,2], summed_data[:,0][:,0])

plt.show()

#create expected results
expected1 = create_expected(data1)
expected2 = create_expected(data2)


print('Input shape:',data1.shape)

print('Output shape')
print(expected1.shape)

print('Creating Model')


#create model
lstm_model=Sequential()

lstm_model.add(TimeDistributed(Dense(200, activation='tanh'), batch_input_shape=(batch_size, tsteps, 3)))

lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(inner,
               batch_input_shape=(batch_size, tsteps, 3),
               return_sequences=True,
               activation='tanh',
               stateful=True))
lstm_model.add(LSTM(inner,
               batch_input_shape=(batch_size, tsteps, 3),
               return_sequences=True,
               activation='tanh',
               stateful=True))

lstm_model.add(LSTM(inner,
               batch_input_shape=(batch_size, tsteps, 3),
               return_sequences=False,
               activation='tanh',
               stateful=True))
lstm_model.add(Dense(3))

lstm_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.RMSprop(lr=learning_rate))

lstm_model.summary()

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    if( i % 2 == 1):
        lstm_model.fit(data1,
                  expected1,
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        lstm_model.reset_states()
    else:
        lstm_model.fit(data2,
                  expected2,
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        lstm_model.reset_states()
        

print('Predicting')
predicted_output = lstm_model.predict(data3, batch_size=batch_size)

print(predicted_output.shape)
print('Ploting Results')


#untransform data and sum
predicted_output = exp_result(predicted_output)
summed_prediction = sum_result_data(predicted_output)


#3D plot
fig = plt.figure()

ax = fig.gca(projection='3d')
print(predicted_output[:,1])
ax.plot(summed_prediction[:,1], summed_prediction[:,2], summed_prediction[:,0])

plt.show()


#animation code
#in seconds
interval = .04

def ani_data_gen():
    global interval
    nextupdate = []
    tick = 0
    for row in summed_prediction:
        
        print(row.shape)
        print(row[0])
        while row[0] > interval*tick:
            print('nothing more on this tick', interval*tick)
            yield np.array(nextupdate)
            nextupdate = []
            tick += 1

        print('adding to tick update', row)
        nextupdate.append((row[1],row[2]))
    
    print('yielding last update')
    yield np.array(nextupdate)
        


fig, ax = plt.subplots()
line, = ax.plot([],[], lw=2)

ax.set_ylim(-2000,2000)
ax.set_xlim(-2000,2000)


xdata = [0]
ydata = [0]

def ani_run(nextupdate):
    global xdata, ydata
    global line
    print('input column slice', nextupdate.shape)
    if nextupdate.shape[0] == 0:
        return line,
    
    xdata = xdata + list(nextupdate[:,0])
    ydata = ydata + list(nextupdate[:,1])
    
    if len(xdata) > 50:
        xdata = xdata[-50:]
    if len(ydata) > 50:
        ydata = ydata[-50:]
    
    line.set_data(xdata,ydata)

    return line,


ani = animation.FuncAnimation(fig, ani_run, ani_data_gen, blit=True, interval=40, repeat=False)

plt.show()
