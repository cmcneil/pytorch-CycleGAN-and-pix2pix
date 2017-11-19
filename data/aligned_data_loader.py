import random
import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from data.np_folder import NpFolder
# from pdb import set_trace as st
# pip install future --upgrade
from builtins import object
from cone.image import conformal


class PairedData(object):
    def __init__(self, data_loader, fineSize, max_dataset_size):
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        # st()

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration

        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)

        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        B = AB[:, :, h_offset:h_offset + self.fineSize,
               w + w_offset:w + w_offset + self.fineSize]

        return {'A': A, 'A_paths': AB_paths, 'B': B, 'B_paths': AB_paths}


class LoaderIter(object):
    def __init__(self, loader, fine_size, max_dataset_size):
        self.loader = loader
        self.max_dataset_size = max_dataset_size
        self.fine_size = fine_size

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        return next(self.loader_iter)


class AlignedNpDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize

        # Dataset A
        if opt.warp_to_square:
            conformal_mapper = conformal.FGSquircularMapper(res=128)
        else:
            conformal_mapper = None
        dataset = NpFolder(root=opt.dataroot + '/' + opt.phase,
                           input_name='mri.npy', label_name='im.npy',
                           return_paths=True, input_nc=opt.input_nc,
                           conformal_mapper=conformal_mapper, opt=opt)
        print '.......Length of dataset: ' + str(len(dataset))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=True,
            num_workers=int(self.opt.nThreads))

        data_loader = data_loader
        self.dataset = dataset
        self.loader_iter = LoaderIter(data_loader, opt.fineSize,
                                      opt.max_dataset_size)

    def name(self):
        return 'AlignedNpDataLoader'

    def load_data(self):
        return self.loader_iter

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


class AlignedDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Scale(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        # Dataset A
        dataset = ImageFolder(root=opt.dataroot + '/' + opt.phase,
                              transform=transform, return_paths=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        self.dataset = dataset
        self.paired_data = PairedData(data_loader, opt.fineSize, opt.max_dataset_size)

    def name(self):
        return 'AlignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
