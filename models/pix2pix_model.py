import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import custom_loss as cust
from cone.image import conformal

from IPython.core.debugger import set_trace

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc,
                                          opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('-----------------------------------------------')

        self.conformal_mapper = conformal.FGSquircularMapper(res=128)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        # print "input A: " + str(self.input_A.size())
        # print self.input_A
        # print "input B: " + str(self.input_B.size())
        # print self.input_B
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # print '...Input size:'
        # print self.input_A.size()
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    #get image paths
    # def get_image_paths(self):
    #     return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())  # fake_AB.detach()
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)#.detach()
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        # print '...real A: ' + str(self.real_A)
        # print '...fake B: ' + str(self.fake_B)
        # print '...real B: ' + str(self.real_B)
        # set_trace()
        # self.conv_dimred = self.netG.children().next().children().next()
        # print self.conv_dimred.weight
        # self.laplace_loss = cust.laplacian_loss(self.conv_dimred.weight)
        # self.orthoreg_loss = cust.orthoreg_loss(self.conv_dimred.weight,
        #                                         gpu_ids=self.gpu_ids)
        # print 'ganloss: ' + str(self.opt.ganloss)
        self.loss_G = (self.opt.ganloss * self.loss_G_GAN + self.loss_G_L1)
                    #    + self.opt.laploss*self.laplace_loss
                    #    + self.opt.orthoregloss*self.orthoreg_loss)
        # print self.orthoreg_loss
                    #    + self.opt.orthoregloss*self.orthoreg_loss)
        # self.loss_G = self.loss_G_L1 + self.opt.laploss*self.laplace_loss

        self.loss_G.backward()

    def optimize_parameters(self, only_d=False):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        if only_d:
            return

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([
                ('G_GAN', self.opt.ganloss*self.loss_G_GAN.data[0]),
                ('G_L1', self.opt.lambda_A*self.loss_G_L1.data[0]),
                ('D_real', self.opt.ganloss*self.loss_D_real.data[0]),
                ('D_fake', self.opt.ganloss*self.loss_D_fake.data[0]),
                # ('G_Laplace', self.opt.laploss*self.laplace_loss.data[0]),
                # ('G_Orthoreg', self.orthoreg_loss.data[0]),
        ])

    def get_current_visuals(self):

        real_A = util.tensor2np(self.real_A.data)

        print '...Shape real A: ' + str(np.shape(real_A))
        return OrderedDict([('real_A', real_A)])

    def get_current_ims(self, whole_batch=False):
        fake_B = util.tensor2np(self.fake_B.data, whole_batch=whole_batch)
        # self.conformal_mapper.square_to_disk(
        real_B = util.tensor2np(self.real_B.data, whole_batch=whole_batch)
        return OrderedDict([('fake_B', fake_B), ('real_B', real_B)])

    def get_filters(self):
        # filts = self.conv_dimred.weight[0].cpu().float()
        # print torch.squeeze(self.conv_dimred.weight).cpu()
        filts = torch.squeeze(self.conv_dimred.weight).cpu().float().data.numpy()
        # print filts
        # print filts.size()
        return filts.T

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def load(self, epoch_label):
        self.load_network(self.netG, 'G', epoch_label)
        self.load_network(self.netD, 'D', epoch_label)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
