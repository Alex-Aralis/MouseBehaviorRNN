'''
A program to animate a cursor movement dataset. 
'''

import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv

batch_size = 100

infile = open('datasets/newinput.csv', 'r')

instream = csv.reader(infile)

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



print('Generating Data')
data = gen_data(instream, batch_size)

print(data.shape)

lastrow = np.array(((0,0,0),))
summed_data = list()
for row in data:
    lastrow = np.add(lastrow,row)
    summed_data.append(lastrow)

summed_data = np.array(summed_data)

#in seconds
interval = .04

def data_gen():
    global interval
    nextupdate = []
    tick = 0
    for row in summed_data[:,0]:
        
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
