import imp
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
from models.normalize import RAIN
from torch.nn.utils import spectral_norm

from models.drconv import DRConv2d,myDRC
from models.deeplab import DeepLabBB
from models.dfconv import DeformConv2d
from models.attention import DualAttention
from models.vgg_arch import VGGFeatureExtractor
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    elif norm_type.startswith('rain'):
        norm_layer = RAIN
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Discriminator1(nn.Module):
    def __init__(self, input_nc, ngf=64): #改了use_attention=True
        super(Discriminator1, self).__init__()
        ngf=16
        input_nc=3
        self.cov1=nn.Sequential(nn.Conv2d(input_nc,ngf,4,2,1),nn.LeakyReLU(0.2,True))#128
        self.cov2=nn.Sequential(nn.Conv2d(ngf,ngf*2,4,2,1),nn.InstanceNorm2d(ngf*2),nn.LeakyReLU(0.2,True))#64
        self.cov3=nn.Sequential(nn.Conv2d(ngf*2,ngf*4,4,2,1),nn.InstanceNorm2d(ngf*4),nn.LeakyReLU(0.2,True))#32
        self.cov4=nn.Sequential(nn.Conv2d(ngf*4,ngf*8,4,2,1),nn.InstanceNorm2d(ngf*8),nn.LeakyReLU(0.2,True))#16
        self.cov5=nn.Sequential(nn.Conv2d(ngf*8,ngf*8,4,2,1),nn.InstanceNorm2d(ngf*8),nn.LeakyReLU(0.2,True))#8
        self.cov6=nn.Sequential(nn.Conv2d(ngf*8,ngf*8,4,2,1),nn.InstanceNorm2d(ngf*8),nn.LeakyReLU(0.2,True))#4
        self.cov7=nn.Sequential(nn.Conv2d(ngf*8,ngf*8,4,2,1),nn.InstanceNorm2d(ngf*8),nn.LeakyReLU(0.2,True))#2
        # self.cov7_1=nn.Sequential(nn.Conv2d(ngf*8,ngf*8,4,2,1),nn.InstanceNorm2d(ngf*8),nn.LeakyReLU(0.2,True))######## 1
        # self.cov7_2=nn.Sequential(nn.Conv2d(ngf*8,ngf*8,4,2,1),nn.InstanceNorm2d(ngf*8),nn.LeakyReLU(0.2,True))###############
        self.cov8=nn.Conv2d(ngf*8,1,2,1)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        x=self.cov1(x)
        x=self.cov2(x)
        x=self.cov3(x)
        x=self.cov4(x)
        x=self.cov5(x)
        x=self.cov6(x)
        x=self.cov7(x)
        # x=self.cov7_1(x)
        # x=self.cov7_2(x)
        x=self.cov8(x)
        x=self.sig(x)
        return x




def define_D1(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    net=Discriminator1(input_nc, ndf)


    return init_net(net, init_type, init_gain, gpu_ids)
#input_nc=3 output_nc=3 ngf=32 netG=hdnet normG=RAIN  dropout=true init_type=normal opt.init_gain=0.02 gpu_ids=0
def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False,
             init_type='normal', init_gain=0.02, gpu_ids=[]):
    """load a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: rainnet
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'hdnet':
        # 3 3 32 norm_layer=RAIN use_dropout=true use_attention=true
        net = HDNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_attention=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'target_decay':
        def target_decay(epoch):
            lr_l = 1
            if epoch >= 100 and epoch < 110:
                lr_l = 0.1
            elif epoch >= 110:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=target_decay)        
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.relu = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean() # self.relu(1-prediction.mean())
            else:
                loss = prediction.mean() # self.relu(1+prediction.mean())
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0, mask=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.unsqueeze(2).unsqueeze(3)
            alpha = alpha.expand_as(real_data)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, mask, gp=True)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def get_act_conv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.Conv2d(dims_in, dims_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*conv)

def get_act_dconv(act, dims_in, dims_out, kernel, stride, padding, bias):
    conv = [act]
    conv.append(nn.ConvTranspose2d(dims_in, dims_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*conv)

class HDNet(nn.Module):
    # 3 3 32 norm_layer=RAIN use_dropout=true use_attention=true
    #改自己的模型
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=RAIN, 
                 norm_type_indicator=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                 use_dropout=False, use_attention=True): #改了use_attention=True
        super(HDNet, self).__init__()
        # self.MYIN=MYIN(ngf,32,16)#区域匹配模块
        # self.NEWMYIN=NEWMYIN(ngf,32,16)
        self.GDCNN=GDCNN(ngf)#多尺度特征融合
        self.a = nn.Conv2d(4, 3, kernel_size=1)
        # self.att1=simam_module()
        # self.att2=simam_module()
        # self.att3=simam_module()
        # self.att4=simam_module()
        # self.myDRC=myDRC(ngf, ngf, 1, 2)
        # self.def_conv1 = nn.Sequential(nn.ReLU(True), DeformConv2d(ngf*8, ngf*8, 3, 1, 1))#原来的
        # self.def_conv2 = nn.Sequential(nn.ReLU(True), DeformConv2d(ngf*8, ngf*8, 3, 1, 1))原来的
        # self.def_conv3 = nn.Sequential(nn.ReLU(True), DeformConv2d(ngf*8, ngf*8, 3, 1, 1))原来的

        ###########残差+simam+动态区域卷积##########
        # self.simam = simam_module()
        # self.drc=DRConv2d(ngf * 4, ngf * 4, 1, 2)

        ################
        ########跳转连接 动态卷积###########
        self.skip1=DRConv2d(ngf, ngf, 1, 2)
        self.skip2=DRConv2d(ngf * 2, ngf * 2, 1, 2)
        self.skip3=DRConv2d(ngf * 4, ngf * 4, 1, 2)
        ###########################

        self.cbam1 = DualAttention(ngf )
        self.cbam2 = DualAttention(ngf*2)
        self.cbam3 = DualAttention(ngf*4)

        ###################语义模型############################
        ''' 
        self.backbone=DeepLabBB(-1, 256, 'resnet34', 0.1)


        self.mask_conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True)
        )
        '''
        #####################################################



        self.input_nc = input_nc
        self.norm_namebuffer = ['RAIN']
        self.use_dropout = use_dropout
        # self.use_attention = use_attention 原来
        self.use_attention = False #改
        ##########
        print("input_nc",input_nc)
        print("ngf",ngf)
        ###########

        # norm_type_list = [get_norm_layer('instance'), norm_layer]原来的
        norm_type_list = [get_norm_layer('instance'), get_norm_layer('instance')]#改
        # -------------------------------Network Settings-------------------------------------
        self.model_layer0         = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=False)
        self.model_layer1         = get_act_conv(nn.LeakyReLU(0.2, True), ngf, ngf*2, 4, 2, 1, False)
        self.model_layer1norm     = norm_type_list[norm_type_indicator[0]](ngf*2)  #注释 我要改为区域归一化
        # self.model_layer1norm=RN_binarylabel(ngf*2)

        self.model_layer2         = get_act_conv(nn.LeakyReLU(0.2, True), ngf*2, ngf*4, 3, 1, 1, False)
        # self.model_layer2         = get_act_conv(nn.LeakyReLU(0.2, True), ngf*2, ngf*4, 4, 2, 1, False)####################################### 1024
        self.model_layer2norm     = norm_type_list[norm_type_indicator[1]](ngf*4) #注释 我要改为区域归一化
        # self.model_layer2norm=RN_binarylabel(ngf*4)

        self.model_layer3         = get_act_conv(nn.LeakyReLU(0.2, True), ngf*4, ngf*8, 4, 2, 1, False)
        self.model_layer3norm     = norm_type_list[norm_type_indicator[2]](ngf*8)
        
        self.model_layer4 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 3, 1, 1, False)
        # self.model_layer4 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 4, 2, 1, False) ####################################### 1024
        # self.model_layer4 = get_act_conv(nn.LeakyReLU(0.2, True), 384, ngf*8, 3, 1, 1, False)###对于语义加入时的情况
        self.model_layer4norm     = norm_type_list[norm_type_indicator[3]](ngf*8)
        
        self.model_layer5 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 4, 2, 1, False)
        self.model_layer5norm  = norm_type_list[norm_type_indicator[4]](ngf*8)

        self.model_layer6 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 3, 1, 1, False)
        # self.model_layer6 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 4, 2, 1, False)####################################### 1024
        self.model_layer6norm  = norm_type_list[norm_type_indicator[5]](ngf*8)
        
        self.model_layer71 = get_act_conv(nn.LeakyReLU(0.2, True), ngf*8, ngf*8, 4, 2, 1, False)
########################### 1024 插入#######################################
        # self.model_layer71norm = nn.InstanceNorm2d(ngf*8)
        # self.model_layer71_1 = get_act_conv(nn.LeakyReLU(0.2, True), ngf * 8, ngf * 8, 3, 1, 1, False)
        # self.model_layer71_1norm = nn.InstanceNorm2d(ngf*8)
        # self.model_layer71_2 = get_act_conv(nn.LeakyReLU(0.2, True), ngf * 8, ngf * 8, 4, 2, 1, False)
        #
        # self.model_layer72_1 = get_act_dconv(nn.ReLU(True), ngf * 8, ngf * 8, 4, 2, 1, False)
        # self.model_layer72_1norm = nn.InstanceNorm2d(ngf*8)
        # self.model_layer72_2 = get_act_dconv(nn.ReLU(True), ngf * 8, ngf * 8, 3, 1, 1, False)
        # self.model_layer72_2norm = nn.InstanceNorm2d(ngf*8)
###################################################################
        self.model_layer72 = get_act_dconv(nn.ReLU(True), ngf*8, ngf*8, 4, 2, 1, False)
        self.model_layer72norm    = norm_type_list[norm_type_indicator[7]](ngf*8)
        
        # self.model_layer8 = get_act_dconv(nn.ReLU(True), ngf*16, ngf*8, 3, 1, 1, False)原来
        self.model_layer8 = get_act_dconv(nn.ReLU(True), ngf*8, ngf*8, 3, 1, 1, False)#改 ####################################### 1024
        # self.model_layer8 = get_act_dconv(nn.ReLU(True), ngf*8, ngf*8, 4, 2, 1, False)#
        self.model_layer8norm    = norm_type_list[norm_type_indicator[8]](ngf*8)
        
        # self.model_layer9 = get_act_dconv(nn.ReLU(True), ngf*16, ngf*8, 4, 2, 1, False)原来
        self.model_layer9 = get_act_dconv(nn.ReLU(True), ngf*8, ngf*8, 4, 2, 1, False)#改
        self.model_layer9norm    = norm_type_list[norm_type_indicator[9]](ngf*8)
        
        # self.model_layer10 = get_act_dconv(nn.ReLU(True), ngf*16, ngf*8, 3, 1, 1, False)原来
        self.model_layer10 = get_act_dconv(nn.ReLU(True), ngf*8, ngf*8, 3, 1, 1, False)#改 ####################################### 1024
        # self.model_layer10 = get_act_dconv(nn.ReLU(True), ngf*8, ngf*8, 4, 2, 1, False)#改
        self.model_layer10norm    = norm_type_list[norm_type_indicator[10]](ngf*8)
        
        if self.use_attention:
            #self.model_layer10att = nn.Sequential(nn.Conv2d(ngf*16, ngf*16, kernel_size=1, stride=1), nn.Sigmoid())
            self.model_layer10att = DRConv2d(ngf*16, ngf*16, 1, 2)
            
        # self.model_layer11        = get_act_dconv(nn.ReLU(True), ngf*16, ngf*4, 4, 2, 1, False)原来
        self.model_layer11        = get_act_dconv(nn.ReLU(True), ngf*8, ngf*4, 4, 2, 1, False)#改
        self.model_layer11norm    = norm_type_list[norm_type_indicator[11]](ngf*4)##注释 我要改为区域归一化
        # self.model_layer11norm=RN_binarylabel(ngf*4)


        if self.use_attention:
            #self.model_layer11att = nn.Sequential(nn.Conv2d(ngf*8, ngf*8, kernel_size=1, stride=1), nn.Sigmoid())
            self.model_layer11att = DRConv2d(ngf*8, ngf*8, 1, 2)
            
        # self.model_layer12        = get_act_dconv(nn.ReLU(True), ngf*8, ngf*2, 3, 1, 1, False)原来
        self.model_layer12        = get_act_dconv(nn.ReLU(True), ngf*4, ngf*2, 3, 1, 1, False) #改 ####################################### 1024
        # self.model_layer12        = get_act_dconv(nn.ReLU(True), ngf*4, ngf*2, 4, 2, 1, False) #改
        self.model_layer12norm    = norm_type_list[norm_type_indicator[12]](ngf*2) ##注释 我要改为区域归一化
        # self.model_layer12norm=RN_binarylabel(ngf*2)

        if self.use_attention:
            #self.model_layer12att = nn.Sequential(nn.Conv2d(ngf*4, ngf*4, kernel_size=1, stride=1), nn.Sigmoid())
            self.model_layer12att = DRConv2d(ngf*4, ngf*4, 1, 2)
            

        # self.model_layer13        = get_act_dconv(nn.ReLU(True), ngf*4, ngf, 4, 2, 1, False)原来
        self.model_layer13        = get_act_dconv(nn.ReLU(True), ngf*2, ngf, 4, 2, 1, False)#改
        self.model_layer13norm    = norm_type_list[norm_type_indicator[13]](ngf)#注释 我要改为区域归一化
        # self.model_layer13norm=RN_binarylabel(ngf)#区域归一化


        if self.use_attention:
            #self.model_layer13att = nn.Sequential(nn.Conv2d(ngf*2, ngf*2, kernel_size=1, stride=1), nn.Sigmoid())
            self.model_layer13att = DRConv2d(ngf*2, ngf*2, 1, 2)
            
        # self.model_out = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1), nn.Tanh())原来
        self.model_out = nn.Sequential(nn.ReLU(True), nn.ConvTranspose2d(ngf , output_nc, kernel_size=3, stride=1, padding=1), nn.Tanh())#改
        # self.cbam = DualAttention(ngf*8)# 原来的
        # self.def_conv = nn.Sequential(nn.ReLU(True), DeformConv2d(ngf*8, ngf*8, 3, 1, 1))原来的

    def forward(self, x, mask):


        x0 = self.model_layer0(x)
        #############
        temp_x=self.GDCNN(x0)
        ############
        x1 = self.model_layer1(temp_x) #输出 [16, 32, 128, 128]
###########################归一化层#####################################################
        if self.model_layer1norm._get_name() in self.norm_namebuffer:
            x1 = self.model_layer1norm(x1, mask)
        else:
            x1 = self.model_layer1norm(x1)


        x2 = self.model_layer2(x1)#[16, 64, 128, 128]
###########################归一化层#####################################################
        if self.model_layer2norm._get_name() in self.norm_namebuffer:
            x2 = self.model_layer2norm(x2, mask)
        else:
            x2 = self.model_layer2norm(x2)


        x3 = self.model_layer3(x2)#[16, 128, 64, 64]
        if self.model_layer3norm._get_name() in self.norm_namebuffer:
            x3 = self.model_layer3norm(x3, mask)
        else:
            x3 = self.model_layer3norm(x3)

        ########语义########
        ''' 
        backbone_mask_features=self.mask_conv(x)
        backbone_features = self.backbone(x[:,:3,:,:], mask, backbone_mask_features).pop()
        #backbone_features.size()#[16, 256, 64, 64]
        x3_1=torch.cat([x3,  backbone_features], 1)
        '''
        #################

        x4 = self.model_layer4(x3)
        if self.model_layer4norm._get_name() in self.norm_namebuffer:
            x4 = self.model_layer4norm(x4, mask)
        else:
            x4 = self.model_layer4norm(x4)

        x5 = self.model_layer5(x4)
        # x5 = self.def_conv3(x5)
        if self.model_layer5norm._get_name() in self.norm_namebuffer:
            x5 = self.model_layer5norm(x5, mask)
        else:
            x5 = self.model_layer5norm(x5)

        x6 = self.model_layer6(x5)
        # x6 = self.def_conv2(x6)
        if self.model_layer6norm._get_name() in self.norm_namebuffer:
            x6 = self.model_layer6norm(x6, mask)
        else:
            x6 = self.model_layer6norm(x6)

        x71 = self.model_layer71(x6)

        # x71 = self.cbam(x71)#原来的
        # x71 = self.def_conv(x71)原来的
        # x71 = self.def_conv1(x71)
        # ########################### 1024 插入#######################################
        # x71= self.model_layer71norm(x71)
        # x71_1=self.model_layer71_1(x71)
        # x71_1=self.model_layer71_1norm(x71_1)
        # x71_2=self.model_layer71_2(x71_1)
        #
        # x72_1=self.model_layer72_1(x71_2)
        # x72_1=self.model_layer72_1norm(x72_1)
        # x72_1= x71_1+x72_1
        # x72_2=self.model_layer72_2(x72_1)
        # x72_2=self.model_layer72_2norm(x72_2)
        # x71= x71+x72_2
        ###################################################################


        x72 = self.model_layer72(x71)
        if self.model_layer72norm._get_name() in self.norm_namebuffer:
            x72 = self.model_layer72norm(x72, mask)
        else:
            x72 = self.model_layer72norm(x72)
        
        # x72 = torch.cat([x6, x72], 1)原来
        x72 = x6+ x72#改
        ###############跳转连接
        '''
        mask1 = F.interpolate(mask.detach(), size=x72.size()[2:], mode='nearest')
        x72 = x6 + x72*mask1
        '''

        ox5 = self.model_layer8(x72)
        if self.model_layer8norm._get_name() in self.norm_namebuffer:
            ox5 = self.model_layer8norm(ox5, mask)
        else:
            ox5 = self.model_layer8norm(ox5)      

        # ox5 = torch.cat([x5, ox5], 1)原来
        ox5 = x5+ ox5#改
        ###############跳转连接
        ''' 
        mask1 = F.interpolate(mask.detach(), size=ox5.size()[2:], mode='nearest')
        ox5  = x5 + ox5  * mask1
        '''

        ox4 = self.model_layer9(ox5)
        if self.model_layer9norm._get_name() in self.norm_namebuffer:
            ox4 = self.model_layer9norm(ox4, mask)
        else:
            ox4 = self.model_layer9norm(ox4)        
        # ox4 = torch.cat([x4, ox4], 1)原来
        ox4 = x4+ ox4#改
        ###############跳转连接
        '''
        mask1 = F.interpolate(mask.detach(), size=ox4.size()[2:], mode='nearest')
        ox4 = x4 + ox4 * mask1
        '''

        ox3 = self.model_layer10(ox4)
        if self.model_layer10norm._get_name() in self.norm_namebuffer:
            ox3 = self.model_layer10norm(ox3, mask)
        else:
            ox3 = self.model_layer10norm(ox3)        
        # ox3 = torch.cat([x3, ox3], 1)  原来
        ox3 = x3+ ox3 #改
        ###############跳转连接
        ''' 
        mask1 = F.interpolate(mask.detach(), size=ox3.size()[2:], mode='nearest')
        ox3 = x3 + ox3 * mask1
        '''

        if self.use_attention:
            ox3 = self.model_layer10att(ox3, mask) 
            #ox3 = self.model_layer10att(ox3) * ox3 

        # #################残差+注意力 动态卷积 第四############################
        # sima=self.simam(ox3)
        # sima=self.drc(sima,mask)
        # ox3=ox3+sima
        # #####################################################

        ox2 = self.model_layer11(ox3)
        ###########################归一化层#####################################################
        if self.model_layer11norm._get_name() in self.norm_namebuffer:
            ox2 = self.model_layer11norm(ox2, mask)
        else:
            ox2 = self.model_layer11norm(ox2)
        # ox2 = self.model_layer11norm(ox2, mask)#对于区域实例化


        ##############第三 跳转连接加入动态卷积###
        # x2=self.skip3(x2, mask)
        ############################
        # ox2 = torch.cat([x2, ox2], 1)原来
        ox2 = x2+ox2#改
        ###############跳转连接

        # mask1 = F.interpolate(mask.detach(), size=ox2.size()[2:], mode='nearest')
        '''
        ox2 = x2 + ox2 * mask1
        '''

        if self.use_attention:
            ox2 = self.model_layer11att(ox2, mask)
            #ox2 = self.model_layer11att(ox2) * ox2 
        # ox2=self.att3(ox2)
        ox2 = self.cbam3(ox2)
        ox2 = self.skip3(ox2, mask)

        # #################残差+注意力 动态卷积 第三############################
        # sima=self.simam(ox2)
        # sima=self.drc(sima,mask)
        # ox2=ox2+sima
        # #####################################################




        ox1 = self.model_layer12(ox2)
        ###########################归一化层#####################################################
        if self.model_layer12norm._get_name() in self.norm_namebuffer:
            ox1 = self.model_layer12norm(ox1, mask)
        else:
            ox1 = self.model_layer12norm(ox1)
        # ox1 = self.model_layer12norm(ox1, mask)#对于区域实例化

        ##############第2 跳转连接加入动态卷积###
        # x1 = self.skip2(x1, mask)
        #####################################
        # ox1 = torch.cat([x1, ox1], 1)原来
        ox1 = x1+ox1#改
        ###############跳转连接

        # mask1 = F.interpolate(mask.detach(), size=ox1.size()[2:], mode='nearest')
        ''' 
        ox1 = x1 + ox1 * mask1
        '''

        if self.use_attention:
            ox1 = self.model_layer12att(ox1, mask) 
            #ox1 = self.model_layer12att(ox1) * ox1
        # ox1=self.att2(ox1)
        ox1 = self.cbam2(ox1)
        ox1=self.skip2(ox1, mask)


        ox0 = self.model_layer13(ox1)
        ###########################归一化层#####################################################
        if self.model_layer13norm._get_name() in self.norm_namebuffer:
            ox0 = self.model_layer13norm(ox0, mask)
        else:
            ox0 = self.model_layer13norm(ox0)
        # ox0 = self.model_layer13norm(ox0, mask)#对于区域实例化

        ##############第1 跳转连接加入动态卷积###
        # x0 = self.skip1(x0, mask)
        #####################################
        # ox0 = torch.cat([x0, ox0], 1)原来
        ox0 = x0+ ox0#改
        ###############跳转连接

        # mask1 = F.interpolate(mask.detach(), size=ox0.size()[2:], mode='nearest')
        ''' 
        ox0 = x0 + ox0 * mask1
        '''

        if self.use_attention:
            ox0 = self.model_layer13att(ox0, mask) 
            #ox0 = self.model_layer13att(ox0) * ox0
        # ox0=self.att1(ox0)
        ox0=self.cbam1(ox0)
        ox0=self.skip1(ox0, mask)

        # ox0=self.myDRC(ox0,mask)
        # ox0 = self.NEWMYIN(ox0, mask)
        # ox0=self.att4(ox0)

        # ox0=self.MYIN(ox0, mask)# 关闭上个归一化层
        out1 = self.model_out(ox0)

        #门控的机制重建层

        aa = torch.sigmoid(self.a(x))

        out=out1*aa+x[:,:3,:,:]*(1-aa)
        
        return out,out1 #训练的
        # return out1,out, #得残差


    def processImage(self, x, mask, background=None):
        if background is not None:
            x = x*mask + background * (1 - mask)
        if self.input_nc == 4:
            x = torch.cat([x, mask], 1) # (bs, 4, 256, 256)
        pred,pred1= self.forward(x, mask)

        return pred * mask  + x[:,:3,:,:] * (1 - mask), pred1 * mask + x[:,:3,:,:] * (1 - mask)

class UnetBlockCodec(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=RAIN, use_dropout=False, use_attention=False, enc=True, dec=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetBlockCodec) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
            enc (bool) -- if use give norm_layer in encoder part.
            dec (bool) -- if use give norm_layer in decoder part.
        """
        super(UnetBlockCodec, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        self.norm_namebuffer = ['RAIN', 'RAIN_Method_Learnable', 'RAIN_Method_BN']
        if outermost:
            self.down = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.submodule = submodule
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        elif innermost:
            self.up = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            )
            self.upnorm = norm_layer(outer_nc) if dec else get_norm_layer('instance')(outer_nc)
        else:
            self.down = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            )
            self.downnorm = norm_layer(inner_nc) if enc else get_norm_layer('instance')(inner_nc)
            self.submodule = submodule
            self.up = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            )
            self.upnorm = norm_layer(outer_nc) if dec else get_norm_layer('instance')(outer_nc)
            if use_dropout:
                self.dropout = nn.Dropout(0.5)

        if use_attention:
            attention_conv = nn.Conv2d(outer_nc+input_nc, outer_nc+input_nc, kernel_size=1)
            attention_sigmoid = nn.Sigmoid()
            self.attention = nn.Sequential(*[attention_conv, attention_sigmoid])

    def forward(self, x, mask):
        if self.outermost:
            x = self.down(x)
            x = self.submodule(x, mask)
            ret = self.up(x)
            return ret
        elif self.innermost:
            ret = self.up(x)
            if self.upnorm._get_name() in self.norm_namebuffer:
                ret = self.upnorm(ret, mask)
            else:
                ret = self.upnorm(ret)
            ret = torch.cat([x, ret], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret
        else:
            ret = self.down(x)
            if self.downnorm._get_name() in self.norm_namebuffer:
                ret = self.downnorm(ret, mask)
            else:
                ret = self.downnorm(ret)
            ret = self.submodule(ret, mask)
            ret = self.up(ret)
            if self.upnorm._get_name() in self.norm_namebuffer:
                ret = self.upnorm(ret, mask)
            else:
                ret = self.upnorm(ret)
            if self.use_dropout:    # only works for middle features
                ret = self.dropout(ret)
            ret = torch.cat([x, ret], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class OrgDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d, global_stages=0):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(OrgDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 0
        self.conv1 = spectral_norm(PartialConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        if global_stages < 1:
            self.conv1f = spectral_norm(PartialConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        else:
            self.conv1f = self.conv1
        self.relu1 = nn.LeakyReLU(0.2, True)
        nf_mult = 1
        nf_mult_prev = 1

        n = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv2 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm2 = norm_layer(ndf * nf_mult)
        if global_stages < 2:
            self.conv2f = spectral_norm(
                PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.norm2f = norm_layer(ndf * nf_mult)
        else:
            self.conv2f = self.conv2
            self.norm2f = self.norm2

        self.relu2 = nn.LeakyReLU(0.2, True)

        n = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv3 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm3 = norm_layer(ndf * nf_mult)
        if global_stages < 3:
            self.conv3f = spectral_norm(
                PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            self.norm3f = norm_layer(ndf * nf_mult)
        else:
            self.conv3f = self.conv3
            self.norm3f = self.norm3
        self.relu3 = nn.LeakyReLU(0.2, True)

        n = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.norm4 = norm_layer(ndf * nf_mult)
        self.conv4 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv4f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm4f = norm_layer(ndf * nf_mult)

        self.relu4 = nn.LeakyReLU(0.2, True)

        n = 4
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv5 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv5f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm5 = norm_layer(ndf * nf_mult)
        self.norm5f = norm_layer(ndf * nf_mult)
        self.relu5 = nn.LeakyReLU(0.2, True)

        n = 5
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv6 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.conv6f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
        self.norm6 = norm_layer(ndf * nf_mult)
        self.norm6f = norm_layer(ndf * nf_mult)
        self.relu6 = nn.LeakyReLU(0.2, True)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.conv7 = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        self.conv7f = spectral_norm(
            PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))

    def forward(self, input, mask=None):
        x = input
        x, _ = self.conv1(x)
        x = self.relu1(x)
        x, _ = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x, _ = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x, _ = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x, _ = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x, _ = self.conv6(x)
        x = self.norm6(x)
        x = self.relu6(x)
        x, _ = self.conv7(x)

        """Standard forward."""
        xf, xb = input, input
        mf, mb = mask, 1 - mask

        xf, mf = self.conv1f(xf, mf)
        xf = self.relu1(xf)
        xf, mf = self.conv2f(xf, mf)
        xf = self.norm2f(xf)
        xf = self.relu2(xf)
        xf, mf = self.conv3f(xf, mf)
        xf = self.norm3f(xf)
        xf = self.relu3(xf)
        xf, mf = self.conv4f(xf, mf)
        xf = self.norm4f(xf)
        xf = self.relu4(xf)
        xf, mf = self.conv5f(xf, mf)
        xf = self.norm5f(xf)
        xf = self.relu5(xf)
        xf, mf = self.conv6f(xf, mf)
        xf = self.norm6f(xf)
        xf = self.relu6(xf)
        xf, mf = self.conv7f(xf, mf)

        xb, mb = self.conv1f(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2f(xb, mb)
        xb = self.norm2f(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3f(xb, mb)
        xb = self.norm3f(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4f(xb, mb)
        xb = self.norm4f(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5f(xb, mb)
        xb = self.norm5f(xb)
        xb = self.relu5(xb)
        xb, mb = self.conv6f(xb, mb)
        xb = self.norm6f(xb)
        xb = self.relu6(xb)
        xb, mb = self.conv7f(xb, mb)

        return x, xf, xb


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        num_outputs = ndf * min(2 ** n_layers, 8)
        self.D = OrgDiscriminator(input_nc, ndf, n_layers, norm_layer)
        self.convl1 = spectral_norm(nn.Conv2d(num_outputs, num_outputs, kernel_size=1, stride=1))
        self.relul1 = nn.LeakyReLU(0.2)
        self.convl2 = spectral_norm(nn.Conv2d(num_outputs, num_outputs, kernel_size=1, stride=1))
        self.relul2 = nn.LeakyReLU(0.2)
        self.convl3 = nn.Conv2d(num_outputs, 1, kernel_size=1, stride=1)
        self.convg3 = nn.Conv2d(num_outputs, 1, kernel_size=1, stride=1)

    def forward(self, input, mask=None, gp=False, feat_loss=False):

        x, xf, xb = self.D(input, mask)
        feat_l, feat_g = torch.cat([xf, xb]), x
        x = self.convg3(x)

        sim = xf * xb
        sim = self.convl1(sim)
        sim = self.relul1(sim)
        sim = self.convl2(sim)
        sim = self.relul2(sim)
        sim = self.convl3(sim)
        sim_sum = sim
        if not gp:
            if feat_loss:
                return x, sim_sum, feat_g, feat_l
            return x, sim_sum
        return (x + sim_sum) * 0.5

class GDCNN(nn.Module):
    def __init__(self,out_channels):
        super(GDCNN, self).__init__()
        self.conv_att = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', dilation=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', dilation=2)
        self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', dilation=4)
        self.conv_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', dilation=6)
        self.cont = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
    def forward(self, x):
        output_att = torch.sigmoid(self.conv_att(x))
        output = torch.cat((self.conv_1(x), self.conv_2(x), self.conv_3(x),self.conv_4(x)), dim=1)
        output = self.cont(output)
        output = x * (1 - output_att) + output * output_att
        return output

class MYIN(nn.Module):
    def __init__(self,dims_in,size,stride,eps=1e-5):
        super(MYIN, self).__init__()
        self.size=size
        self.stride=stride
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        ###############################################################################
        self.q_gamma=nn.Parameter(torch.ones(dims_in), requires_grad=True)
        self.q_beta=nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.k_gamma=nn.Parameter(torch.ones(dims_in), requires_grad=True)
        self.k_beta=nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        ###############################################################################
        self.unfold=nn.Unfold(kernel_size=size, stride=stride,padding=0)
        self.fold = nn.Fold(kernel_size=size, stride=stride,padding=0,output_size=(256,256))

        #o=[(i+2p-k)/s]+1
        #线性函数 用于分块背景
        self.nums=(256-size)//stride+1
        self.att=att(self.nums,self.nums)
        self.eps = eps

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')

        mean_back, std_back = self.get_foreground_mean_std(x * (1 - mask), 1 - mask)  # the background features
        normalized = ((x - mean_back) / std_back)*(1-mask)#实例化背景区域norm
        #normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) +self.background_beta[None, :, None, None]) * (1 - mask)
        normalized_background = (x * (1 + self.background_gamma[None, :, None, None]) +self.background_beta[None, :, None, None]) * (1 - mask)

        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask)
        normalized_foreground=((x - mean_fore) / std_fore)*mask     #实例化前景

        #获取前景块 背景块mask块——————————————————————————————————————————————————
        B,C,H,W=x.size()

        count = torch.ones(1, 1, H, W).to(torch.device('cuda:0'))#用于统计每个部分被利用了几次
        #归一化前 背景块
        patch_back = self.unfold(x * (1 - mask)).view(B, C, self.size ** 2, -1).permute(0, 1, 3,2).contiguous()
        #计算用于QKV
        normalized_patch_fore=self.unfold(normalized_foreground).view(B, C, self.size ** 2, -1).permute(0, 1, 3, 2).contiguous()#IN后前景块
        #
        normalized_patch_back = self.unfold(normalized).view(B, C, self.size ** 2, -1).permute(0, 1, 3, 2).contiguous()#IN后背景块
        patch_mask=self.unfold(mask).view(B,1,self.size**2,-1).permute(0, 1,3, 2).contiguous()##IN后mask块  B C 块数 每个块的个数
        count = self.unfold(count)
        #计算块norm后值
        patch_back,patch_back_std,normalized_patch_fore_mean,normalized_patch_back=self.get_patch_mean_std(patch_back,normalized_patch_fore,normalized_patch_back,patch_mask)#计算块内前景和背景的均值
        #进行特征调整
        normalized_patch_fore_mean=normalized_patch_fore_mean*self.q_gamma[None, :, None]+self.q_beta[None, :, None]
        normalized_patch_back=normalized_patch_back*self.k_gamma[None, :, None]+self.k_beta[None, :, None]
        #利用注意力求解特征均值和平方均值
        patch_back,patch_back_std=self.att(normalized_patch_fore_mean,normalized_patch_back,patch_back,patch_back_std)

        ##调整后前景块
        normalized_patch_fore=normalized_patch_fore*patch_back_std+patch_back

        normalized_patch_fore=normalized_patch_fore.permute(0, 1,3, 2).contiguous().view(B,C*(self.size**2),-1).contiguous()
        count=self.fold(count)
        normalized_foreground=self.fold(normalized_patch_fore)
        normalized_foreground=torch.div(normalized_foreground,count)


        #实例化后的前景加入参数

        normalized_foreground = (normalized_foreground * (1 + self.foreground_gamma[None, :, None, None]) +self.foreground_beta[None, :, None, None]) * mask
        #normalized_foreground+x*(1-mask)
        return normalized_foreground+normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])  # (B, C)
        num = torch.sum(mask, dim=[2, 3])  # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]

        var = torch.sum((region + (1 - mask) * mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]


        return mean, torch.sqrt(var + self.eps)
    def get_patch_mean_std(self, region1, region3,region4,mask):
        #patch_back,normalized_patch_fore,normalized_patch_back,patch_mask
        sum = torch.sum(region1, dim=[3])  # (B, C,patchnorm)
        num = torch.sum(1-mask, dim=[3])  # (B, C,patchnorm)
        mu1 = sum / (num + self.eps)
        std = torch.sum((region1 + mask* mu1[:,:,:,None] - mu1[:,:,:,None]) ** 2, dim=[3]) / (num + self.eps)

        sum = torch.sum(region4, dim=[3])
        mu4 = sum / (num + self.eps)
        ##############################
        num = torch.sum(mask, dim=[3])
        sum = torch.sum(region3, dim=[3])
        mu3 = sum / (num + self.eps)
        return mu1,std,mu3,mu4
class MYIN1(nn.Module):
    def __init__(self,dims_in,size,stride,eps=1e-5):
        super(MYIN1, self).__init__()
        self.size=size
        self.stride=stride
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        ###############################################################################
        # self.q_gamma=nn.Parameter(torch.ones(dims_in), requires_grad=True)
        # self.q_beta=nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        # self.k_gamma=nn.Parameter(torch.ones(dims_in), requires_grad=True)
        # self.k_beta=nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        ###############################################################################
        self.unfold=nn.Unfold(kernel_size=size, stride=stride,padding=0)
        self.fold = nn.Fold(kernel_size=size, stride=stride,padding=0,output_size=(256,256))
        self.liner_fore=nn.Conv2d(dims_in, dims_in, kernel_size=(1,self.size**2), bias=False)
        self.liner_back=nn.Conv2d(dims_in, dims_in, kernel_size=(1,self.size**2), bias=False)

        #o=[(i+2p-k)/s]+1
        #线性函数 用于分块背景
        self.nums=(256-size)//stride+1
        self.att=att(self.nums,self.nums)
        self.eps = eps

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')

        mean_back, std_back = self.get_foreground_mean_std(x * (1 - mask), 1 - mask)  # the background features
        normalized = ((x - mean_back) / std_back)*(1-mask)#实例化背景区域norm
        #normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) +self.background_beta[None, :, None, None]) * (1 - mask)
        normalized_background = (x * (1 + self.background_gamma[None, :, None, None]) +self.background_beta[None, :, None, None]) * (1 - mask)

        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask)
        normalized_foreground=((x - mean_fore) / std_fore)*mask     #实例化前景

        #获取前景块 背景块mask块——————————————————————————————————————————————————
        B,C,H,W=x.size()

        count = torch.ones(1, 1, H, W).to(torch.device('cuda:0'))#用于统计每个部分被利用了几次
        #归一化前 背景块
        patch_back = self.unfold(x * (1 - mask)).view(B, C, self.size ** 2, -1).permute(0, 1, 3,2).contiguous()
        #计算用于QKV
        normalized_patch_fore=self.unfold(normalized_foreground).view(B, C, self.size ** 2, -1).permute(0, 1, 3, 2).contiguous()#IN后前景块
        #
        normalized_patch_back = self.unfold(normalized).view(B, C, self.size ** 2, -1).permute(0, 1, 3, 2).contiguous()#IN后背景块
        patch_mask=self.unfold(mask).view(B,1,self.size**2,-1).permute(0, 1,3, 2).contiguous()##IN后mask块  B C 块数 每个块的个数
        count = self.unfold(count)
        #计算块norm后值
        patch_back,patch_back_std,normalized_patch_fore_mean,normalized_patch_back_mean=self.get_patch_mean_std(patch_back,normalized_patch_fore,normalized_patch_back,patch_mask)#计算块内前景和背景的均值
        ##########改#############
        IN_patch_back = normalized_patch_back*(1-patch_mask) + normalized_patch_back_mean[:, :, :, None] * patch_mask
        IN_patch_fore = normalized_patch_fore*patch_mask + normalized_patch_fore_mean[:, :, :, None] * (1 - patch_mask)

        token_back = self.liner_back(IN_patch_back).view(B, C, -1).contiguous()  # B C 块数量 1 ->B C 块数量
        token_fore = self.liner_fore(IN_patch_fore).view(B, C, -1).contiguous()  # B C 块数量 1 ->B C 块数量

        #######################

        #进行特征调整
        # token_fore=token_fore*self.q_gamma[None, :, None]+self.q_beta[None, :, None]
        # token_back=token_back*self.k_gamma[None, :, None]+self.k_beta[None, :, None]

        #利用注意力求解特征均值和平方均值
        patch_back,patch_back_std=self.att(token_fore,token_back,patch_back,patch_back_std)

        ##调整后前景块
        normalized_patch_fore=normalized_patch_fore*patch_back_std+patch_back

        normalized_patch_fore=normalized_patch_fore.permute(0, 1,3, 2).contiguous().view(B,C*(self.size**2),-1).contiguous()
        count=self.fold(count)
        normalized_foreground=self.fold(normalized_patch_fore)
        normalized_foreground=torch.div(normalized_foreground,count)


        #实例化后的前景加入参数

        normalized_foreground = (normalized_foreground * (1 + self.foreground_gamma[None, :, None, None]) +self.foreground_beta[None, :, None, None]) * mask
        #normalized_foreground+x*(1-mask)
        return normalized_foreground+normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])  # (B, C)
        num = torch.sum(mask, dim=[2, 3])  # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]

        var = torch.sum((region + (1 - mask) * mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var + self.eps)
    def get_patch_mean_std(self, region1, region3,region4,mask):
        #patch_back,normalized_patch_fore,normalized_patch_back,patch_mask
        sum = torch.sum(region1, dim=[3])  # (B, C,patchnorm)
        num = torch.sum(1-mask, dim=[3])  # (B, C,patchnorm)
        mu1 = sum / (num + self.eps)
        std = torch.sum((region1 + mask* mu1[:,:,:,None] - mu1[:,:,:,None]) ** 2, dim=[3]) / (num + self.eps)

        sum = torch.sum(region4, dim=[3])
        mu4 = sum / (num + self.eps)
        ##############################
        num = torch.sum(mask, dim=[3])
        sum = torch.sum(region3, dim=[3])
        mu3 = sum / (num + self.eps)
        return mu1,std,mu3,mu4

class NEWMYIN(nn.Module):
    def __init__(self,dims_in,size,stride,eps=1e-5):
        super(NEWMYIN, self).__init__()
        self.size=size
        self.stride=stride
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        ###############################################################################
        # self.q_gamma=nn.Parameter(torch.ones(dims_in), requires_grad=True)
        # self.q_beta=nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        # self.k_gamma=nn.Parameter(torch.ones(dims_in), requires_grad=True)
        # self.k_beta=nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        ###############################################################################
        self.unfold=nn.Unfold(kernel_size=size, stride=stride,padding=0)
        self.fold = nn.Fold(kernel_size=size, stride=stride,padding=0,output_size=(256,256))
        self.liner_fore=nn.Conv2d(dims_in, dims_in, kernel_size=(1,self.size**2), bias=False)
        self.liner_back=nn.Conv2d(dims_in, dims_in, kernel_size=(1,self.size**2), bias=False)

        #o=[(i+2p-k)/s]+1
        #线性函数 用于分块背景
        self.nums=(256-size)//stride+1
        self.att=att(self.nums,self.nums)
        self.eps = eps

    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        invmask=1-mask
        background=x * invmask
        foreground=x*mask
        #背景均值 标准差
        # back_mean, back_std = self.get_foreground_mean_std(background, invmask)  # the background features
        #背景不标准化
         # normalized = ((x - back_mean) / back_std) * invmask  # 实例化背景区域norm
        # normalized_background = (normalized * (1 + self.background_gamma[None, :, None, None]) +self.background_beta[None, :, None, None]) * (1 - mask)
        unormalized_background = (background * (1 + self.background_gamma[None, :, None, None]) + self.background_beta[None, :,
                                                                                        None, None]) * invmask
        #获取 前景块 背景块 mask块
        B, C, H, W = x.size()
        count = torch.ones(1, 1, H, W).to(torch.device('cuda:0'))  # 用于统计每个部分被利用了几次
        patch_back = self.unfold(background).view(B, C, self.size ** 2, -1).permute(0, 1, 3, 2).contiguous() #B C 块大小 块数量->B C 数量 块大小
        patch_fore = self.unfold(foreground).view(B, C, self.size ** 2, -1).permute(0, 1, 3,2).contiguous()  # 前景块
        patch_mask = self.unfold(mask).view(B, 1, self.size ** 2, -1).permute(0, 1, 3,2).contiguous()  ##mask块  B C 块数 每个块的个数
        #背景块 和前景块标准化
        patch_back_mean,patch_back_std=self.patch_mean_std(patch_back,1-patch_mask)# B C 块数
        patch_fore_mean,patch_fore_std=self.patch_mean_std(patch_fore,patch_mask)# B C 块数
        IN_patch_back=((patch_back - patch_back_mean[:,:,:,None]) / patch_back_std[:,:,:,None])*(1-patch_mask)
        IN_patch_fore =((patch_fore - patch_fore_mean[:,:,:,None]) / patch_fore_std[:,:,:,None])*patch_mask
        #将0替换上
        IN_patch_back=IN_patch_back+patch_back_mean[:,:,:,None]*patch_mask
        IN_patch_fore=IN_patch_fore+patch_fore_mean[:,:,:,None]*(1-patch_mask)
        #IN背景块 IN前景块 线性投影
        token_back=self.liner_back(IN_patch_back).view(B, C, -1).contiguous()#B C 块数量 1 ->B C 块数量
        token_fore=self.liner_fore(IN_patch_fore).view(B, C, -1).contiguous()#B C 块数量 1 ->B C 块数量
        # token_back = token_back * self.k_gamma[None, :, None] + self.k_beta[None, :, None]
        # token_fore = token_fore * self.q_gamma[None, :, None] + self.q_beta[None, :, None]

        #计算前景token和背景token 注意力
        new_mean,new_std=self.att(token_fore,token_back,patch_back_mean,patch_back_std)
        #IN前景块 用新的特征
        new_patch_fore=IN_patch_fore*new_std+new_std #B C 数量 块大小
        #重构前景
        new_foreground = new_patch_fore.permute(0, 1, 3, 2).contiguous().view(B, C * (self.size ** 2),
                                                                                -1).contiguous()
        count = self.unfold(count)
        count = self.fold(count)
        new_foreground = self.fold(new_foreground)
        new_foreground = torch.div(new_foreground, count)
        new_foreground = (new_foreground * (
                    1 + self.foreground_gamma[None, :, None, None]) + self.foreground_beta[None, :, None, None]) * mask

        return new_foreground+unormalized_background

    def patch_mean_std(self, region,mask):
        #patch_back,normalized_patch_fore,normalized_patch_back,patch_mask
        sum = torch.sum(region, dim=[3])  # (B, C,patchnorm)
        num = torch.sum(mask, dim=[3])  # (B, C,patchnorm)
        mu = sum / (num + self.eps)
        std = torch.sum((region -mask* mu[:,:,:,None] ) ** 2, dim=[3]) / (num + self.eps)
        return mu,std
    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])  # (B, C)
        num = torch.sum(mask, dim=[2, 3])  # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]

        var = torch.sum((region + (1 - mask) * mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var + self.eps)

class att(nn.Module):
    def __init__(self,h,w):
        super(att, self).__init__()
        self.relative_positive_bias_table = nn.Parameter(torch.zeros((2*h-1)*(2*w-1)), requires_grad=True)
        self.sm = nn.Softmax(dim=-1)
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += h - 1
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)



    def forward(self, q,k,v,std):
        '''
        q；实例化前景
        k:实例化背景
        v:未实例化的背景特征
        v1patch均方值
        v2patch平方均方值
        '''
        # print("____________")
        # print(q.size())
        #print(k.size())
        # print(v.size())
        # self.relative_positive_bias_table.view: -> [Mh*Mw*Mh*Mw, num_head] -> [Mh*Mw, Mh*Mw, num_head]
        B, C, L = q.size()  # 前景的

        relative_position_bias = self.relative_positive_bias_table[self.relative_position_index.view(-1)].view(
            L, L).contiguous()


        Q = q.view(B, C, L).permute(0, 2, 1).contiguous()  # B H*W C
        K = k.view(B, C,L).contiguous()#.to(torch.device('cuda:2')) #to(torch.device('cuda:1'))# B C H1*W1
        # S = self.sm(torch.bmm(Q, K)/(C**(-0.5))+relative_position_bias[None,:,:])#计算出attmap
        S = self.sm(torch.bmm(Q, K)+relative_position_bias[None,:,:])#计算出attmap
        #S = self.sm(torch.bmm(Q, K) )
        V=v.view(B, C, L).permute(0, 2, 1).contiguous()
        std = std.view(B, C, L).permute(0, 2, 1).contiguous()#.to(torch.device('cuda:2'))
        V= torch.bmm(S, V).view(B, L, C).permute(0, 2, 1).contiguous()  # B HW C
        std=torch.bmm(S, std).view(B, L, C).permute(0,2,1).contiguous()

        return V[:,:,:,None],std[:,:,:,None]

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights= {'conv5_4': 1},
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        # self.criterion_type = criterion
        self.criterion_type = 'l2'
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()

        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()#L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)

        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0

            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]+percep_loss
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

class MSE(nn.Module):
    def __init__(self, min_area=100):
        super(MSE, self).__init__()
        self.min_area = min_area

    def forward(self, pred, label):
        loss = (pred - label) ** 2
        reduce_dims = (1, 2, 3)
        loss = torch.mean(loss, dim=reduce_dims)
        loss = torch.sum(loss) / pred.size(0)
        return loss

class RN_binarylabel(nn.Module):
    def __init__(self, feature_channels):
        super(RN_binarylabel, self).__init__()
        self.eps = 1e-5

    def forward(self, x, label):
        '''
        input:  x: (B,C,M,N), features
                label: (B,1,M,N), 1 for foreground regions, 0 for background regions
        output: _x: (B,C,M,N)
        '''
        label = F.interpolate(label.detach(), size=x.size()[2:], mode='nearest')

        rn_foreground_region = self.rn(x * label, label)

        rn_background_region = self.rn(x * (1 - label), 1 - label)

        return rn_foreground_region + rn_background_region

    def rn(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])  # (B, C,patchnorm)
        num = torch.sum(mask, dim=[2, 3])  # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask) * mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        var= torch.sqrt(var + self.eps)
        region = ((region - mean) / var) * mask
        return region
