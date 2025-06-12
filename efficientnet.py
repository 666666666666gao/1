import torch
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
import torch.nn as nn
import torch.nn.functional as F
from mobileVitblock import MobileViTBlock

def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue 
    g = gaussian_filter(n,sigma)
    gaussiankernel = torch.from_numpy(g)
    
    return gaussiankernel.float()

def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue
    l = laplace(n)
    laplacekernel = torch.from_numpy(l)
    
    return laplacekernel.float()


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class SEM(nn.Module):
    def __init__(self, ch_out, reduction=None):
        super(SEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out//reduction, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch_out//reduction, ch_out, kernel_size=1,bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        
        return x * y.expand_as(x)

class EEM(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, groups, reduction):
        super(EEM, self).__init__()
        
        self.groups = groups
        self.gk = gaussiankernel(ch_in, int(ch_in/groups), kernel, kernel-2, 0.9)
        self.lk = laplaceiankernel(ch_in, int(ch_in/groups), kernel, 0.9)
        self.gk = nn.Parameter(self.gk, requires_grad=False)
        self.lk = nn.Parameter(self.lk, requires_grad=False)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.05),
            nn.InstanceNorm2d(int(ch_out/2))
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=1,padding=0,groups=2),
            nn.PReLU(num_parameters=int(ch_out/2), init=0.05),
            nn.InstanceNorm2d(int(ch_out/2))
            )

        self.conv3 = nn.Sequential(
                    nn.MaxPool2d(3, stride=1, padding=1),
                    nn.Conv2d(int(ch_out/2), ch_out, kernel_size=1,padding=0,groups=2),
                    nn.PReLU(num_parameters=ch_out, init=0.01),
                    nn.GroupNorm(4, ch_out)
                    )
            
        self.sem1 = SEM(ch_out, reduction=reduction)  # let's put reduction as 2
        self.sem2 = SEM(ch_out, reduction=reduction)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.03)
      
    def forward(self, x):
        DoG = F.conv2d(x, self.gk.to('cuda'), padding='same',groups=self.groups)
        LoG = F.conv2d(DoG, self.lk.to('cuda'), padding='same',groups=self.groups)
        DoG = self.conv1(DoG-x)
        LoG = self.conv2(LoG)
        tot = self.conv3(DoG*LoG)
        
        tot1 = self.sem1(tot)
        x1 = self.sem2(x)
        
        return self.prelu(x+x1+tot+tot1)


class SubSpace(nn.Module):
    """
    Subspace class.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int) -> None:
        super(SubSpace, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out


class ULSAM(nn.Module):
    """
    Grouped Attention Block having multiple (num_splits) Subspaces.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    nout : int
        number of output feature maps

    h : int
        height of a input feature map

    w : int
        width of a input feature map

    num_splits : int
        number of subspaces

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int, nout: int, h: int, w: int, num_splits: int) -> None:
        super(ULSAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.h = h
        self.w = w
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out


class EfficientCrackNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(EfficientCrackNet, self).__init__()
        self.maxpool = nn.MaxPool2d(2,2)

        self.DSC1 = SeparableConv2d(in_channels=3, out_channels=16, kernel_size=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.EEM1 = EEM(ch_in=16, ch_out=16, kernel=3, groups=1, reduction=2)

        self.DSC2 = SeparableConv2d(in_channels=16, out_channels=32, kernel_size=3, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.EEM2 = EEM(ch_in=32, ch_out=32, kernel=3, groups=1, reduction=2)

        self.DSC3 = SeparableConv2d(in_channels=32, out_channels=64, kernel_size=3, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.EEM3 = EEM(ch_in=64, ch_out=64, kernel=3, groups=1, reduction=2)
        self.ULSAM1 = ULSAM(64, 64, 14, 28, 4)

        self.DSC4 = SeparableConv2d(in_channels=64, out_channels=16, kernel_size=3, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(16)
        self.MobileViTBlock1 = MobileViTBlock(dim=32, depth=3, channel=16, kernel_size=3, patch_size=(2,2), mlp_dim=int(32*2))

        self.DSC5 = SeparableConv2d(in_channels=16, out_channels=32, kernel_size=3, bias=False)
        self.batch_norm5 = nn.BatchNorm2d(32)
        self.MobileViTBlock2 = MobileViTBlock(dim=64, depth=3, channel=32, kernel_size=3, patch_size=(2,2), mlp_dim=int(64*2))

        self.DSC6 = SeparableConv2d(in_channels=32, out_channels=64, kernel_size=3, bias=False)
        self.batch_norm6 = nn.BatchNorm2d(64)

        self.final_conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.final_ulsam = ULSAM(32, 32, 24, 34, 4)
        self.final_conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0)

        self.upsampling1 = nn.Upsample(scale_factor=2)
        self.DSC7 = SeparableConv2d(in_channels=128, out_channels=64, kernel_size=3, bias=False)
        self.batch_norm7 = nn.BatchNorm2d(64)

        self.upsampling2 = nn.Upsample(scale_factor=2)
        self.DSC8 = SeparableConv2d(in_channels=96, out_channels=32, kernel_size=3, bias=False)
        self.batch_norm8 = nn.BatchNorm2d(32)
        self.ULSAM2 = ULSAM(32, 32, 48, 68, 4)

        # self.upsampling3 = nn.Upsample(scale_factor=2)
        self.DSC9 = SeparableConv2d(in_channels=48, out_channels=16, kernel_size=3, bias=False)
        self.batch_norm9 = nn.BatchNorm2d(16)
        self.MobileViTBlock3 = MobileViTBlock(dim=32, depth=3, channel=16, kernel_size=3, patch_size=(2,2), mlp_dim=int(32*2))

        self.upsampling4 = nn.Upsample(scale_factor=2)
        self.DSC10 = SeparableConv2d(in_channels=80, out_channels=64, kernel_size=3, bias=False)
        self.batch_norm10 = nn.BatchNorm2d(64)
        self.ULSAM3 = ULSAM(64, 64, 48, 68, 4)

        self.upsampling5 = nn.Upsample(scale_factor=2)
        self.DSC11 = SeparableConv2d(in_channels=96, out_channels=32, kernel_size=3, bias=False)
        self.batch_norm11 = nn.BatchNorm2d(32)
        self.ULSAM4 = ULSAM(32, 32, 96, 136, 4)

        self.DSC12 = SeparableConv2d(in_channels=48, out_channels=16, kernel_size=3, bias=False)
        self.batch_norm12 = nn.BatchNorm2d(16)

        self.last_block = nn.Conv2d(16, output_ch, kernel_size=1,bias=False)


    def forward(self, x):
        # Encoder Block 1
        encoder_block1_x = F.relu(self.batch_norm1(self.DSC1(x)))
        encoder_block1_out = self.EEM1(encoder_block1_x)
        # print('encoder_block1_out:', encoder_block1_out.shape)
        # Encoder Block 2
        encoder_block2_x = F.relu(self.batch_norm2(self.DSC2(encoder_block1_out)))
        encoder_block2_out = self.EEM2(encoder_block2_x)
        # print('encoder_block2_out:', encoder_block2_out.shape)
        # Encoder Block 3
        encoder_block3_x = self.maxpool(F.relu(self.batch_norm3(self.DSC3(encoder_block2_out))) )
        encoder_block3_out = self.ULSAM1(self.EEM3(encoder_block3_x))
        # print('encoder_block3_out:', encoder_block3_out.shape)
        # Encoder Block 4
        encoder_block4_x = self.maxpool(F.relu(self.batch_norm4(self.DSC4(encoder_block3_out))) )
        encoder_block4_out = self.MobileViTBlock1(encoder_block4_x)
        # print('encoder_block4_out:', encoder_block4_out.shape)
        # Encoder Block 5
        encoder_block5_x = F.relu(self.batch_norm5(self.DSC5(encoder_block4_out)))
        encoder_block5_out = self.MobileViTBlock2(encoder_block5_x)
        # print('encoder_block5_out:', encoder_block5_out.shape)
        # Encoder Block 6
        encoder_block6_x = self.maxpool(F.relu(self.batch_norm6(self.DSC6(encoder_block5_out))) )
        # Output to be reversed
        encoder_output = self.final_conv2(self.final_ulsam(self.final_conv1(encoder_block6_x)))
        # Decoder Block 1
        decoder_block1_x = self.upsampling1(encoder_output)
        decoder_block1_x_ = torch.cat((encoder_block6_x, decoder_block1_x), dim=1)
        decoder_block1_out = F.relu(self.batch_norm7(self.DSC7(decoder_block1_x_)))
        # print('decoder_block1_out:', decoder_block1_out.shape)
        # Decoder Block 2
        decoder_block2_x = self.upsampling2(decoder_block1_out)
        decoder_block2_x_ = torch.cat((encoder_block5_out, decoder_block2_x), dim=1)
        decoder_block2_out = self.ULSAM2(F.relu(self.batch_norm8(self.DSC8(decoder_block2_x_))))
        # Decoder Block 3
        decoder_block3_x = decoder_block2_out
        decoder_block3_x_ = torch.cat((encoder_block4_out, decoder_block3_x), dim=1)
        decoder_block3_out = self.MobileViTBlock3(F.relu(self.batch_norm9(self.DSC9(decoder_block3_x_))))
        # Decoder Block 4
        decoder_block4_x = self.upsampling4(decoder_block3_out)
        decoder_block4_x_ = torch.cat((encoder_block3_out, decoder_block4_x), dim=1)
        decoder_block4_out = self.ULSAM3(F.relu(self.batch_norm10(self.DSC10(decoder_block4_x_))))
        # Decoder Block 5
        decoder_block5_x = self.upsampling5(decoder_block4_out)
        decoder_block5_x_ = torch.cat((encoder_block2_out, decoder_block5_x), dim=1)
        decoder_block5_out = self.ULSAM4(F.relu(self.batch_norm11(self.DSC11(decoder_block5_x_))))
        # Decoder Block 6
        decoder_block6_x = decoder_block5_out
        decoder_block6_x_ = torch.cat((encoder_block1_out, decoder_block6_x), dim=1)
        decoder_block6_out = F.relu(self.batch_norm12(self.DSC12(decoder_block6_x_)))
        
        last_out = F.relu(self.last_block(decoder_block6_out))
        return last_out