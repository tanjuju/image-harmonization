
import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
from .fMSE import MaskWeightedMSE
from .networks import PerceptualLoss

class HDNetModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['comp', 'real', 'output', 'mask', 'real_f', 'fake_f', 'bg', 'attentioned']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
            # self.model_names = ['G','D'] #生成对抗网络
        else:
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        # opt.input_nc=3 opt.output_nc=3 opt.ngf=32 opt.netG=hdnet opt.normG=RAIN  opt.no_dropout=FALSE opt.init_type=normal opt.init_gain=0.02
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.relu = nn.ReLU()
        if self.isTrain:

            # define loss functions
            self.criterionL1 = MaskWeightedMSE(100)
            self.criterionL2 = PerceptualLoss()#感知损失
            self.criterionL3 = MaskWeightedMSE(100)#控制损失
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # opt.lr=0.001 opt.g_lr_ratio=1.0 opt.beta1=0.9
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            #############对抗###############
            # netD = networks.Discriminator1(3,16)
            # self.netD=networks.init_net(netD,opt.init_type,opt.init_gain,self.gpu_ids)
            # self.criterionGAN = networks.GANLoss('wgangp').to(self.device)
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * opt.g_lr_ratio,
            #                                     betas=(opt.beta1, 0.999))
            # self.optimizers.append(self.optimizer_D)
            ##############################


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.inputs = self.comp
        if self.opt.input_nc == 4:
            self.inputs = torch.cat([self.inputs, self.mask], 1)  # channel-wise concatenation
        self.real_f = self.real * self.mask
        self.bg = self.real * (1 - self.mask)

    def forward(self):
        self.output,self.output1 = self.netG(self.inputs, self.mask)
        self.fake_f = self.output * self.mask
        self.attentioned = self.output * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
        self.attentioned_init=self.output1 * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
        self.harmonized = self.attentioned
    # def backward_D(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake;
    #     fake_AB = self.harmonized
    #     pred_fake= self.netD(fake_AB.detach())
    #     self.loss_D_fake = pred_fake.mean()
    #
    #     # Real
    #     real_AB = self.real
    #     pred_real = self.netD(real_AB)
    #
    #     self.loss_D_real = - pred_real.mean()
    #     # self.loss_D_global = global_fake + global_real
    #
    #
    #     # gradient_penalty, gradients = networks.cal_gradient_penalty(self.netD, real_AB.detach(), fake_AB.detach(),
    #     #                                                             'cuda', mask=self.mask)
    #     # self.loss_D_gp = gradient_penalty
    #
    #     # combine loss and calculate gradients
    #     self.loss_D = self.loss_D_fake + self.loss_D_real# + self.opt.gp_ratio * gradient_penalty
    #     self.loss_D.backward()
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        #################生成对抗#######################
        # fake_AB = self.harmonized
        # pred_fake= self.netD(fake_AB)
        # self.loss_G_global = self.criterionGAN(pred_fake, True)
        # self.loss_G_GAN = self.loss_G_global
        ##############################################
        self.loss_G_L1 = self.criterionL1(self.attentioned, self.real, self.mask) * self.opt.lambda_L1
        self.loss_G_L2,_ = self.criterionL2(self.attentioned, self.real)#感知损失
        self.loss_G_L3 = self.criterionL1(self.attentioned_init, self.real, self.mask)

        self.loss_G = self.loss_G_L1 +self.loss_G_L2*0.0001+ self.loss_G_L3*0.5
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
##################################
        # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()  # set D's gradients to zero
        # self.backward_D()  # calculate gradients for D
        # self.optimizer_D.step()  # update D's weights
        # self.set_requires_grad(self.netD, False)  # enable backprop for D
   ################################################
         # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

