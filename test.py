import time
import os
import numpy as np
import matplotlib.pyplot as plt

from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html


opt.nThreads = 1   # test code only supports nThreads=1
opt.serial_batches = True  # no shuffle

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    print "... A:" + str(data['A'].size()) + "... B:" + str(data['B'].size())
    model.set_input(data)
    model.test()
    visuals = model.get_current_ims(whole_batch=True)
    print np.shape(visuals)
    for label, np_data in visuals.items():
        print label
        print np.shape(np_data)
        for j in range(np.shape(np_data)[0]):
            if np.shape(np_data[j, ...])[0] == 45:
                np.save('../images/im_' + label + '_' + str(i*opt.batchSize+j) + '.npy',
                        np.reshape(np_data[j, ...], (15, 3, 128, 128)))
            else:
                plt.imsave('../images/im_' + label + '_' + str(i*opt.batchSize+j) + '.png',
                           np_data[j, ...].T, cmap='viridis')
