import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from ssim import SSIM
import time
from .LOG_loss import *

class InfraGAN(BaseModel):
    def name(self):
        return 'InfraGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain # or True,

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          resolution=256 if opt.dataset_mode == 'thermal' else 512)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain: # or True:
                self.load_network(self.netD, 'D', opt.which_epoch)


        self.fake_AB_pool = ImagePool(opt.pool_size)
        # define loss functions
        
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor) # GAN loss  
        self.criterionL1 = torch.nn.L1Loss() # L1 loss
        self.ssim = SSIM() # ssim 
        self.log_loss = LogLoss() # log loss implementation
        
        # initialize optimizers
        if self.isTrain:
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr/100, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.input_B = input_B
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            t = time.time()
            self.fake_B = self.netG(self.real_A)
            t = time.time() - t
            self.real_B = Variable(self.input_B)
            """
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake, pred_middle_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) + self.criterionGAN(pred_middle_fake, True)
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
            self.loss_G = self.loss_G_GAN + self.loss_G_L1

            self.loss_D_fake = self.criterionGAN(pred_fake, False) + self.criterionGAN(pred_middle_fake, False)
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real, pred_middle_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True) + self.criterionGAN(pred_middle_real, True)
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            """
        return t
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake, pred_middle_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) + self.criterionGAN(pred_middle_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real, pred_middle_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True) + self.criterionGAN(pred_middle_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake, pred_middle_fake = self.netD(fake_AB)
        with torch.autograd.set_detect_anomaly(True):
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) + self.criterionGAN(pred_middle_fake, True)

            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
            
            self.loss_ssim = (1-self.ssim(self.fake_B.clone(), self.real_B.clone())) * self.opt.lambda_A
            
            self.loss_log = self.log_loss(self.fake_B,self.real_B) # log loss 
#             print("log  loss fakeB and realB",self.loss_log)
            
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_ssim+self.loss_log
#             self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_ssim
            
            # self.loss_G = self.loss_G_GAN + self.loss_G_L1

            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('G_L1', self.loss_G_L1.data),
                            ('D_real', self.loss_D_real.data),
                            ('D_fake', self.loss_D_fake.data)
                            ])

    @staticmethod
    def get_errors():
        return OrderedDict([('G_GAN', 0),
                            ('G_L1', 0),
                            ('D_real', 0),
                            ('D_fake', 0)
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.thermal_tensor2im(self.fake_B.data)
        real_B = util.thermal_tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

