import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from data.mrf_folder import MRFFolder
# pip install future --upgrade
from builtins import object
from pdb import set_trace as st

import numpy as np

class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A = None
        B = None
        try:
            A = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A = next(self.data_loader_A_iter)

        try:
            B = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'A': A,
                    'B': B}

class UnalignedDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
                                       transforms.Scale(opt.loadSize),
                                       transforms.RandomCrop(opt.fineSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))])

        # Dataset A
        dataset_A = ImageFolder(root=opt.dataroot + '/' + opt.phase + 'A',
                                transform=transform, return_paths=True)
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        # Dataset B
        dataset_B = ImageFolder(root=opt.dataroot + '/' + opt.phase + 'B',
                                transform=transform, return_paths=True)
        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.paired_data = PairedData(data_loader_A, data_loader_B, self.opt.max_dataset_size)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_A), len(self.dataset_B)), self.opt.max_dataset_size)

class UnalignedMRFDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        # transform = transforms.Compose([
        #                                transforms.Scale(opt.loadSize),
        #                                transforms.RandomCrop(opt.fineSize),
        #                                transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5),
        #                                                     (0.5, 0.5, 0.5))])
        subslices = np.load(opt.subslices)

        # Dataset A
        dataset_A = MRFFolder(bucket=opt.bucket_A, bucket_path=opt.dataset_cc_path,
                              subslices=subslices,
                              transform=None, return_paths=False, dset='mrf')
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=30)
            # num_workers=int(self.opt.nThreads))

        # Dataset B
        dataset_B = MRFFolder(bucket=opt.bucket_B, bucket_path=opt.dataset_cc_path,
                              subslices=subslices,
                              transform=None, return_paths=False,
                              dset='quant')
        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=30)
            # num_workers=int(self.opt.nThreads))
        # print "dataloader B: " + str(data_loader_B)
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.paired_data = PairedData(data_loader_A, data_loader_B, self.opt.max_dataset_size)

    def name(self):
        return 'UnalignedMRFDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_A), len(self.dataset_B)), self.opt.max_dataset_size)
