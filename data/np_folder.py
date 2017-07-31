import torch.utils.data as data

import os
import os.path
import numpy as np
import random


class NpFolder(data.Dataset):

    def __init__(self, root=None, input_name=None, label_name=None,
                 transform=None, return_paths=False, input_nc=None):
        """
        """
        print "data root: " + str(root)
        subslices = []
        for l in os.listdir(root):
            for s in os.listdir(os.path.join(root, l)):
                subslices.append(l + '/' + s)
        # for p, d, f in os.walk(root):
        #     if len(d) > 0:
        #         for direc in d:
        #             subslices.append(p.split('/')[-1] + '/' + direc)
        # self.imgs = imgs
        # random.shuffle(subslices)
        self.root = root
        self.subslices = subslices
        self.input_name = input_name
        self.label_name = label_name
        self.return_paths = return_paths
        self.input_nc = input_nc
        # print subslices

    def __getitem__(self, index):
        ss = self.subslices[index]
        # print ss
        path = os.path.join(self.root, ss)
        # print path
        input_path = os.path.join(path, self.input_name)
        output_path = os.path.join(path, self.label_name)

        indata = np.load(input_path)
        if self.input_nc is not None:
            indata = indata[:self.input_nc, ...]
        outdata = np.load(output_path)
        return {'A': indata, 'B': outdata}

    def __len__(self):
        return len(self.subslices)
