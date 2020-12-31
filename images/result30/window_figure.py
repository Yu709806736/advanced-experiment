import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

with open('window_record.csv', 'r') as file:
    name = file.readline().replace(' ', '_')
    _ = file.readline()
    line = file.readline().replace('\n', '')
    x = []
    window_sizes = []
    accs = []
    recs = []
    x_rec = []
    while line:
        line = line.split(', ')
        x.append(float(line[0]))
        window_sizes.append(float(line[1]))
        accs.append(float(line[2]))
        if line[3] != 'n/a':
            recs.append(float(line[3]))
            x_rec.append(float(line[0]))
        line = file.readline().replace('\n', '')

    x = np.arange(0, len(x))
    plt.title("Window size - training steps figure")
    plt.xlabel('training steps')
    plt.ylabel('window size')
    # plt.ylim((1000, 5010))
    plt.plot(x, window_sizes)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(10000))
    plt.savefig('./window_size.png')
    plt.show()
    plt.close()
    print('done1')

    plt.title('accuracy - training steps figure')
    plt.xlabel('training steps')
    plt.ylabel('accuracy')
    plt.plot(x, accs, c='r', label='accuracy')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10000))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.savefig('./accuracy.png')
    plt.show()
    print('done2')

    plt.title('recall - training steps figure')
    plt.plot(x_rec, recs, c='g', label='recall')
    plt.xlabel('training steps')
    plt.ylabel('recall')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10000))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.savefig('./recall.png')
    plt.show()

    accs = np.array(accs)
    recs = np.array(recs)
    print('accuracy average = {0}, variance = {1}'.format(np.mean(accs), np.var(accs)))
    print('recall average = {0}, variance = {1}'.format(np.mean(recs), np.var(recs)))

    sys.exit(0)
