################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################
import torch.utils.data as data

import glabtools.io as io

# def make_dataset(bucket_path, subslice, fnames):
#     """
#     Return list of paths in cottoncandy bucket.
#
#     bucket_path : str
#         The path to the directory that has the individual subject data. Path
#         should have fields to fill in, e.g. "/path/to/{subject}/"
#
#     subslice : list of tuples
#         List of (subject ID (str), slice number (int)) for every slice.
#
#     fnames : list
#         Names of files that should be extracted from each folder.
#     """
#     images = []
#
#     for sspair in subslice:
#         isub, islice = sspair
#         f_ims = []
#         if len(fnames) > 1: # quantitative images
#             for fname in fnames:
#                 path = bucket_path.format(subject=isub, fname=fname)
#                 f_ims.append(path)
#             f_ims.append(islice)
#         else:
#             f_ims.append(bucket_path.format(subject=isub, slice=str(islice), fname=fnames[0]))
#         images.append(f_ims)
#
#     return images


# def default_loader(path):
#     return Image.open(path).convert('RGB')


class MRFFolder(data.Dataset):

    def __init__(self, bucket, bucket_path, subslices, return_paths=False,
                 transform=None, dset='mrf'):
        """
        dset: a string, 'mrf' or 'quant', specifying which dataset is desired.
        """
        # imgs = make_dataset(bucket_path, subslice, fnames)
        if len(subslices) == 0:
            raise(RuntimeError("Found 0 images!!! AHHHHHH :0"))
        # self.imgs = imgs
        self.subslices = subslices
        self.transform = transform
        self.return_paths = return_paths
        self.bucket = bucket
        self.bucket_path = bucket_path
        self.dset = dset

        self.cci = io.get_cc_interface(self.bucket)

    def __getitem__(self, index):
        subject, nslice = self.subslices[index]

        if self.dset == 'mrf':
            path = self.bucket_path.format(subject=subject,
                                           slice=str(nslice),
                                           fname='mrf.npy')
        elif self.dset == 'quant':
            path = self.bucket_path.format(subject=subject,
                                           slice=str(nslice),
                                           fname='t1_t2_pd.npy')
        else:
            raise ValueError('Invalid value for parameter dset.')
        # print('Downloading: ' + str(path))
        dl_tries = 0
        while True:
            try:
                img = self.cci.download_npy_array(path)
                break
            except:
                dl_tries += 1
                if dl_tries > 10:
                    raise Exception('failed to download ' + str(path))
        return img

    def __len__(self):
        return len(self.subslices)
