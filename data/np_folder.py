import torch.utils.data as data

import os
import os.path
import numpy as np
import skimage.transform


class NpFolder(data.Dataset):

    def __init__(self, root=None, input_name=None, label_name=None,
                 scale_size=None, return_paths=False, input_nc=None,
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
        self.scale_size = scale_size

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
        if self.scale_size is not None and self.scale_size != indata.shape[-1]:
            if len(indata.shape) != 3 or len(outdata.shape) != 3:
                raise ValueError("The fineSize option only makes sense when your "
                                 + "data is of shape (channels, width, height)")
            new_in_shape = list(indata.shape)
            new_in_shape[-1] = self.scale_size
            new_in_shape[-2] = self.scale_size
            new_out_shape = list(outdata.shape)
            new_out_shape[-1] = self.scale_size
            new_out_shape[-2] = self.scale_size

            indata = skimage.transform.resize(indata, tuple(new_in_shape))
            outdata = skimage.transform.resize(outdata, tuple(new_out_shape))
        assert np.shape(outdata)[0] == self.opt.output_nc
        return {'A': indata, 'B': outdata}

    def __len__(self):
        return len(self.ims)
