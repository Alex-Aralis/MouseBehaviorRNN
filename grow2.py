'''
This code creates the RNN and then displays result as a 3d
model and an animation of the cursor movemnt.
'''

from __future__ import print_function
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

infile = open('datasets/newinput.csv', 'r')
infile2 = open('datasets/set2.csv', 'r')

instream = csv.reader(infile)
instream2 = csv.reader(infile2)

#for row in instream:
#    print(row)

def gen_data(instream, batch_size):

    raw_data = []
    for row in instream:
        raw_data.append(row)

    data_size = len(raw_data) - len(raw_data) % batch_size

    data = []
    
    for row in raw_data:
        data.append(((float(row[0]),float(row[4]),float(row[5])),))

    data = data[0:len(data)-len(data)%batch_size]

    return np.array(data)


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 100
epochs = 1000
inner = 1000


print('Generating Data')
data = gen_data(instream, batch_size)
data2 = gen_data(instream2, batch_size)

lastrow = np.array(((0,0,0),))
summed_data = list()
for row in data2:
    lastrow = np.add(lastrow,row)
    summed_data.append(lastrow)

summed_data = np.array(summed_data)

fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot(summed_data[:,0][:,1], summed_data[:,0][:,2], summed_data[:,0][:,0])

plt.show()



#print(np.array(data[:,0][:,0]))

x_series = list()
y_series = list()
delta_time_series = list()

for num in data[:,0][:,1]:
    x_series.append(((num,),))
for num in data[:,0][:,2]:
    y_series.append(((num,),))
for num in data[:,0][:,0]:
    delta_time_series.append(((num,),))

#print(delta_time_series)

delta_time_series=np.array(delta_time_series)
x_series = np.array(x_series)
y_series = np.array(y_series)


x_series2 = list()
y_series2 = list()
delta_time_series2 = list()

for num in data2[:,0][:,1]:
    x_series2.append(((num,),))
for num in data2[:,0][:,2]:
    y_series2.append(((num,),))
for num in data2[:,0][:,0]:
    delta_time_series2.append(((num,),))

#print(delta_time_series)

delta_time_series2=np.array(delta_time_series2)
x_series2 = np.array(x_series2)
y_series2 = np.array(y_series2)


#print(delta_time_series, delta_time_series.shape)

expected = np.append(data[1:],data[0:1], axis=0)

tmp = []
for row in expected:
    tmp.append(row[0])

expected = np.array(tmp)


expected2 = np.append(data2[1:],data2[0:1], axis=0)

tmp = []
for row in expected2:
    tmp.append(row[0])

expected2 = np.array(tmp)

#print(expected)
#print(len(expected))


print('Input shape:',data.shape)

print('Output shape')
print(expected.shape)

print('Creating Model')
#model.add(TimeDistributed(Dense(inner, activation='tanh'), batch_input_shape=(batch_size, tsteps, 3)))


'''
x_model = Sequential(name='x_seq_model')
y_model = Sequential(name='y_seq_modle')
delta_time_model = Sequential(name='delta_time_seq_model')



x_model.add(TimeDistributed(Dense(32, input_dim=1), batch_input_shape=(batch_size, tsteps, 1), name='x_TDD'))
y_model.add(TimeDistributed(Dense(32, input_dim=1), batch_input_shape=(batch_size, tsteps, 1), name='y_TDD'))
delta_time_model.add(TimeDistributed(Dense(100), batch_input_shape=(batch_size, tsteps, 1), name='dT_TDD'))
'''

'''
x_pos_in = TimeDistributed(Input(dtype='int32', name='x_in', batch_shape=(batch_size, 1)), batch_input_shape=(batch_size, tsteps, 1), input_shape=(tsteps, 1))
y_pos_in = TimeDistributed(Input(dtype='int32', name='y_in', batch_shape=(batch_size, 1)), batch_input_shape=(batch_size, tsteps, 1), input_shape=(tsteps, 1))
delta_time_in = TimeDistributed(Input(name='delta_time', dtype='float32', batch_shape=(batch_size, 1)), batch_input_shape=(batch_size, tsteps, 1), input_shape=(tsteps,1))

x_pos_in.build((batch_size, tsteps, 1))
y_pos_in.build((batch_size, tsteps, 1

x_embedded = TimeDistributed(Embedding(1920, 64), batch_input_shape=(batch_size, tsteps, 1))(x_pos_in)
y_embedded = TimeDistributed(Embedding(1080, 64), batch_input_shape=(batch_size, tsteps, 1))(y_pos_in)

delta_time_embedded = TimeDistributed(Dense(100), batch_input_shape=(batch_size, tsteps, 1))(delta_time_in)
'''


'''
x_model.summary()
y_model.summary()
delta_time_model.summary()


merged = Merge([x_model, y_model], mode='concat', concat_axis=2)
merged = Merge([delta_time_model, merged], mode='concat', concat_axis=2)
'''

lstm_model=Sequential()

'''
lstm_model.add(merged)
'''

lstm_model.add(TimeDistributed(Dense(200, activation='tanh'), batch_input_shape=(batch_size, tsteps, 3)))


lstm_model.add(LSTM(inner,
               batch_input_shape=(batch_size, tsteps, 3),
               return_sequences=True,
               activation='tanh',
               stateful=True))
lstm_model.add(Dropout(0.2))
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

lstm_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.RMSprop(lr=.001))

lstm_model.summary()

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    if( i % 2 == 1):
        lstm_model.fit(data,
                  expected,
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
predicted_output = lstm_model.predict(data, batch_size=batch_size)

print(predicted_output.shape)
print('Ploting Results')

lastrow = np.array([0,0,0])
summed_prediction = list()

for row in predicted_output:
    lastrow = np.add(np.concat((abs(row[0]), row[1], row[2])),lastrow)
    summed_prediction.append(list(lastrow))

summed_prediction = np.array(summed_prediction)

fig = plt.figure()

ax = fig.gca(projection='3d')
print(predicted_output[:,1])
ax.plot(summed_prediction[:,1], summed_prediction[:,2], summed_prediction[:,0])

plt.show()

#normal end

#in seconds
interval = .04

def data_gen():
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

def run(nextupdate):
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


ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=40, repeat=False)


# Set up formatting for the movie files
'''
Writer = animation.writers['ffmpeg']
writer = Writer(metadata=dict(artist='Me'), bitrate=1800)

ani.save('newinput.mp4')
'''
plt.show()


