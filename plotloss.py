#!/usr/bin/env python

from optparse import OptionParser
import numpy as np
import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import matplotlib.pyplot as plt

def plotloss(logfile, title, ofile):
    with open(logfile,'rb') as f:
        #f.readline()
        lines = f.readlines()
        #random.shuffle(lines)
        data = {"train_time":[],"train_loss":[],"train_error":[],"test_time":[],"test_loss":[],"test_error":[]}
        train_loss = []
        train_error = []
        test_loss = []
        test_error = []
        for line in lines:
            
            if line[0:9]==" | Epoch:":
                info = line[line.index("Time"):].strip()
                info = info.split()
                #print(info)
                if info[7] == 'nan':
                    continue
                #data['train_time'].append(info[1])
                train_loss.append(float(info[5]))
                train_error.append(float(info[7]))
            if line[0:8]==" | Test:":
                info = line[line.index("Time"):].strip()
                info = info.split()
                #print(info)
                if info[7] == 'nan':
                    continue
                #data['test_time'].append(info[1])
                test_loss.append(float(info[5]))
                test_error.append(float(info[7]))
            if line[0:11] == " * Finished":
                data['train_loss'].append(np.mean(train_loss))
                data['train_error'].append(np.mean(train_error))
                data['test_loss'].append(np.mean(test_loss))
                data['test_error'].append(np.mean(test_error))
                
        #print data
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(data['train_loss'],label="Train Loss")
        ax[0].plot(data['test_loss'],label="Test Loss")
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epochs')
        minTestLoss = np.min(data['test_loss'])
        ax[0].set_title('Loss on '+title+"\n Min(test loss)="+str(minTestLoss))
        ax[0].legend(loc='upper center', shadow=True, fontsize='x-large')
        #error
        ax[1].plot(data['train_error'],label="Train Top1 Error")
        ax[1].plot(data['test_error'],label="Test Top1 Error")
        ax[1].set_ylabel('Top1 Error')
        ax[1].set_xlabel('Epochs')
        minTestError = np.min(data['test_error'])
        ax[1].set_title("Top1 Error on "+title+"\n Min(test error)="+str(minTestError))
        ax[1].legend(loc='upper center', shadow=True, fontsize='x-large')
        #plt.show() 
	plt.savefig(ofile)
    	plt.close()
    	print("saved into "+ofile)


if __name__ == "__main__":

    optParser = OptionParser()
    optParser.add_option("-i", "--ifile", action = "store", type = 'string', \
        dest = "ifile", default = "1.log", help = "input filename")
    optParser.add_option("-t", "--title", action = "store", type = 'string', \
        dest = "title", default = "title", help = "description")
    optParser.add_option("-o", "--ofile", action = "store", type = 'string', \
        dest = "ofile", default = "train_log.png", help = "output filename")
    (opt, args) = optParser.parse_args()
    ifile  = opt.ifile
    ofile  = opt.ofile
    title  = opt.title
    plotloss(ifile, title, ofile)

