import torch.utils.data as data

import os
import os.path
import numpy as np


class NpFolder(data.Dataset):

    def __init__(self, root=None, input_name=None, label_name=None,
                 transform=None, return_paths=False, input_nc=None,
                 conformal_mapper=None, opt=None):
        """
        """
        print "data root: " + str(root)
        ims = []
        for n in os.listdir(root):
            ims.append(n)

        self.root = root
        self.ims = ims
        self.input_name = input_name
        self.label_name = label_name
        self.return_paths = return_paths
        self.input_nc = input_nc
        self.conformal_mapper = conformal_mapper
        self.opt = opt
        # print '******************'
        # print "self.ims: " + str(self.ims)
        # print '*********'

    def __getitem__(self, index):
        n = self.ims[index]
        path = os.path.join(self.root, n)
        input_path = os.path.join(path, self.input_name)
        output_path = os.path.join(path, self.label_name)

        indata = np.load(input_path)
        outdata = np.load(output_path)

        if self.conformal_mapper is not None:
            indata = self.conformal_mapper.disk_to_square(indata)
            outdata = self.conformal_mapper.disk_to_square(outdata)
            # print outdata
        assert np.shape(outdata)[0] == self.opt.output_nc
        return {'A': indata, 'B': outdata}

    def __len__(self):
        return len(self.ims)
