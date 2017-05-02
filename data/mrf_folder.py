################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################

import torch.utils.data as data

import os
import os.path
import cottoncandy as cc
from scipy import ndimage



def make_bias_field(t1w, t2w):
    """
    Return bias field image given T1w and T2w images.
    """
    biasf = np.sqrt(t1w * tw2)
    biasf[biasf < 100] = 0
    biasf_smooth = ndimage.gaussian_filter(biasf, sigma=(2, 2, 2))
    biasf_smooth[biasf_smooth < 100] = 10**5
    return biasf_smooth


def make_dataset(bucket_path, subslice, fnames):
    """
    Return list of paths in cottoncandy bucket.

    bucket_path : str
        The path to the directory that has the individual subject data. Path
        should have fields to fill in, e.g. "/path/to/{subject}/"

    subslice : list of tuples
        List of (subject ID (str), slice number (int)) for every slice.

    fnames : list
        Names of files that should be extracted from each folder.
    """
    images = []

    for sspair in subslice:
        isub, islice = sspair
        f_ims = []
        if len(fnames) > 1: # quantitative images
            for fname in fnames:
                path = bucket_path.format(subject=isub, fname=fname)
                f_ims.append(path)
            f_ims.append(islice)
        else:
            f_ims.append(bucket_path.format(subject=isub, slice=str(islice), fname=fname))
        images.append(f_ims)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class MRFFolder(data.Dataset):

    def __init__(self, bucket, bucket_path, subslice, fnames, return_paths=False,
                 loader=None):
        imgs = make_dataset(bucket_paths, subslice, fnames)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images!!! AHHHHHH :0"))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.bucket = bucket
        self.TE = 3.53

    def __getitem__(self, index):
        cci = cc.get_interface(self.bucket)

        paths = self.imgs[index]

        # If there is more than one path -> quant t1, t2; weighted t1, t2
        # use weighted t1/t2 to find proton density and stack with qT1, qT2
        #
        # If there is only one path then it's from the simulation and just
        # return the relevant data.
        if len(paths) > 1:
            islice = paths[-1]
            slices = []

            # First two entries are qT1, qT2
            for path in paths[:2]:
                s = cci.download_npy_array(path)
                slices.append(s[islice, :, :])

            # Get proton density from T1w, T2w
            t1w = cci.download_npy_array(path[2])
            t2w = cci.download_npy_array(path[3])
            biasf = make_bias_field(t1w, t2w)
            t2w_bc = t2w / biasf
            t2w_bc_sub = t2w_bc[90:200, :, :] # ugh hard coding
            # Proton density img
            pd = t2w_bc_sub[islice, :, :] / np.exp(-self.TE / slices[1])
            pd[np.isnan(pd)] = 0
            pd[np.isinf(pd)] = 0
            slices.append(pd)

            img = np.stack(slices, axis=2)
        else:
            img = cci.download_npy_array(paths[0])

        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
return len(self.imgs)
