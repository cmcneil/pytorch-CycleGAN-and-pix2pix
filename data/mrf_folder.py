################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################

import torch.utils.data as data
import cottoncandy as cc

import os
import os.path



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


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

    def __getitem__(self, index):
        cci = cc.get_interface(self.bucket)

        paths = self.imgs[index]

        # Get the relevant slices and stack them into an image
        # if there is more than one path, i.e. there is a qT1 and qT2 volume.
        # If there is only one path then it's from the simulation and just
        # return the relevant data.
        if len(paths) > 1:
            islice = paths[-1]
            slices = []
            for path in paths[:-1]:
                s = cci.download_npy_array(path)
                slices.append([islice, :, :])
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
