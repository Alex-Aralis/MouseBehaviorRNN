'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
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
from keras.layers import Input, Embedding, Merge
import csv
import keras

infile = open('../../../csv-datasets/tfile.csv', 'r')

instream = csv.reader(infile)

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
batch_size = 5000
epochs = 20
inner = 500


print('Generating Data')
data = gen_data(instream, batch_size)

'''
fig = plt.figure()

ax = fig.gca(projection='3d')
print(data)
ax.plot(data[:,0][:,1], data[:,0][:,2], data[:,0][:,0])

plt.show()

'''

print(np.array(data[:,0][:,0]))

x_series = list()
y_series = list()
delta_time_series = list()

for num in data[:,0][:,1]:
    x_series.append(((num,),))
for num in data[:,0][:,2]:
    y_series.append(((num,),))
for num in data[:,0][:,0]:
    delta_time_series.append(((num,),))

print(delta_time_series)

delta_time_series=np.array(delta_time_series)
x_series = np.array(x_series)
y_series = np.array(y_series)

print(delta_time_series, delta_time_series.shape)

expected = np.append(data[1:],data[0:1], axis=0)

tmp = []
for row in expected:
    tmp.append(row[0])

expected = np.array(tmp)

print(expected)
print(len(expected))


print('Input shape:',data.shape)

print('Output shape')
print(expected.shape)

print('Creating Model')
#model.add(TimeDistributed(Dense(inner, activation='tanh'), batch_input_shape=(batch_size, tsteps, 3)))

x_model = Sequential(name='x_seq_model')
y_model = Sequential(name='y_seq_modle')
delta_time_model = Sequential(name='delta_time_seq_model')



x_model.add(TimeDistributed(Dense(32, input_dim=1), batch_input_shape=(batch_size, tsteps, 1), name='x_TDD'))
y_model.add(TimeDistributed(Dense(32, input_dim=1), batch_input_shape=(batch_size, tsteps, 1), name='y_TDD'))
delta_time_model.add(TimeDistributed(Dense(100), batch_input_shape=(batch_size, tsteps, 1), name='dT_TDD'))


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

x_model.summary()
y_model.summary()
delta_time_model.summary()


merged = Merge([x_model, y_model], mode='concat', concat_axis=2)
merged = Merge([delta_time_model, merged], mode='concat', concat_axis=2)


lstm_model=Sequential()

lstm_model.add(merged)

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
               return_sequences=True,
               activation='tanh',
               stateful=True))
lstm_model.add(LSTM(inner,
               batch_input_shape=(batch_size, tsteps, 3),
               return_sequences=False,
               activation='tanh',
               stateful=True))
lstm_model.add(Dense(3))

lstm_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.RMSprop(lr=.01))

lstm_model.summary()

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    lstm_model.fit([delta_time_series, x_series, y_series],
              expected,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    lstm_model.reset_states()

print('Predicting')
predicted_output = lstm_model.predict([delta_time_series, x_series, y_series], batch_size=batch_size)

print(predicted_output.shape)
print('Ploting Results')

fig = plt.figure()

ax = fig.gca(projection='3d')
print(predicted_output[:,1])
ax.plot(predicted_output[:,1], predicted_output[:,2], predicted_output[:,0])

plt.show()
