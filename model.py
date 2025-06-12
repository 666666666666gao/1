import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, laplace

class UNet_FCN(nn.Module):
    def __init__(self, args, scaler=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, int(32/scaler), 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(int(32/scaler))
        self.conv2 = nn.Conv2d(int(32/scaler), int(64/scaler), 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(int(64/scaler))
        self.maxpool = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(int(64/scaler), int(128/scaler), 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(int(128/scaler))
        # maxpool
        self.conv4 = nn.Conv2d(int(128/scaler), int(128/scaler), 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(int(128/scaler))
        # maxpool
        self.conv5 = nn.Conv2d(int(128/scaler), int(256/scaler), 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(int(256/scaler))
        self.conv6 = nn.Conv2d(int(256/scaler), int(256/scaler), 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(int(256/scaler))

        self.conv7 = nn.Conv2d(int(256/scaler), int(512/scaler), 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(int(512/scaler))
        self.conv8 = nn.Conv2d(int(512/scaler), int(512/scaler), 3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(int(512/scaler))

        self.conv9 = nn.Conv2d(int(512/scaler), int(512/scaler), 3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(int(512/scaler))
        self.conv10 = nn.Conv2d(int(512/scaler), int(1024/scaler), 3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(int(1024/scaler))

        self.transconv1 = nn.ConvTranspose2d(int(1024/scaler), int(512/scaler), 2, stride=2)
        self.conv11 = nn.Conv2d(int(1024/scaler), int(512/scaler), 3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(int(512/scaler))
        self.conv12 = nn.Conv2d(int(512/scaler), int(256/scaler), 3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(int(256/scaler))

        self.transconv2 = nn.ConvTranspose2d(int(256/scaler), int(256/scaler), 2, stride=2)
        self.conv13 = nn.Conv2d(int(512/scaler), int(256/scaler), 3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(int(256/scaler))
        self.conv14 = nn.Conv2d(int(256/scaler), int(128/scaler), 3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(int(128/scaler))

        self.transconv3 = nn.ConvTranspose2d(int(128/scaler), int(128/scaler), 2, stride=2)
        self.conv15 = nn.Conv2d(int(256/scaler), int(128/scaler), 3, stride=1, padding=1)
        self.bn15 = nn.BatchNorm2d(int(128/scaler))
        self.conv16 = nn.Conv2d(int(128/scaler), int(64/scaler), 3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(int(64/scaler))

        self.transconv4 = nn.ConvTranspose2d(int(64/scaler), int(64/scaler), 2, stride=2)
        self.conv17 = nn.Conv2d(int(128/scaler), int(64/scaler), 3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm2d(int(64/scaler))
        self.conv18 = nn.Conv2d(int(64/scaler), int(64/scaler), 3, stride=1, padding=1)
        self.bn18 = nn.BatchNorm2d(int(64/scaler))
        self.conv19 = nn.Conv2d(int(64/scaler), 1, 1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        l2 = self.maxpool(x1)

        x2 =  F.relu(self.bn3(self.conv3(l2)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        l5 = self.maxpool(x2)

        x3 = F.relu(self.bn5(self.conv5(l5)))
        x3 = F.relu(self.bn6(self.conv6(x3)))
        l7 = self.maxpool(x3)
        
        x4 = F.relu(self.bn7(self.conv7(l7)))
        x4 = F.relu(self.bn8(self.conv8(x4)))  
        l9 = self.maxpool(x4)

        x5 = F.relu(self.bn9(self.conv9(l9)))
        x5 = F.relu(self.bn10(self.conv10(x5)))  

        trans_x1 = self.transconv1(x5)
        trans_x1 = torch.cat((trans_x1, x4), 1)
        trans_x1 = F.relu(self.bn11(self.conv11(trans_x1)))
        trans_x1 = F.relu(self.bn12(self.conv12(trans_x1)))

        trans_x2 = self.transconv2(trans_x1)
        trans_x2 = torch.cat((trans_x2, x3), 1)
        trans_x2 = F.relu(self.bn13(self.conv13(trans_x2)))
        trans_x2 = F.relu(self.bn14(self.conv14(trans_x2)))

        trans_x3 = self.transconv3(trans_x2)
        trans_x3 = torch.cat((trans_x3, x2), 1)
        trans_x3 = F.relu(self.bn15(self.conv15(trans_x3)))
        trans_x3 = F.relu(self.bn16(self.conv16(trans_x3)))

        trans_x4 = self.transconv4(trans_x3)
        trans_x4 = torch.cat((trans_x4, x1), 1)
        trans_x4 = F.relu(self.bn17(self.conv17(trans_x4)))
        trans_x4 = F.relu(self.bn18(self.conv18(trans_x4)))

        final_x = self.sigmoid(self.conv19(trans_x4))

        return final_x
    
# Ohter models I want to implement: Mask RCNN, Segformer 


def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue): # sigma = 2, kernel value = 0.9
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


class FRCM(nn.Module):
    def __init__(self,ch_ins,ch_out,n_sides=11):
        super(FRCM,self).__init__()

        self.reducers = nn.ModuleList([
            nn.Conv2d(ch_ins[0],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[1],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[2],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[3],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[4],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[5],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[6],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[7],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[8],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[9],ch_out,kernel_size=1),
            nn.Conv2d(ch_ins[10],ch_out,kernel_size=1)
            ])

        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.1)
        self.gn = nn.GroupNorm(1, ch_out)
        
        self.fused = nn.Conv2d(ch_out*n_sides, ch_out, kernel_size=1)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.1)
        
        for m in self.reducers:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fused.weight, std=0.01)
        nn.init.constant_(self.fused.bias, 0)

    def get_weight(self):
        return [self.fused.weight]
    
    def get_bias(self):
        return [self.fused.bias]

    def forward_sides(self, sides, img_shape):
        # pass through base_model and store intermediate activations (sides)
        late_sides = []
        for x, conv in zip(sides, self.reducers):
            x = F.interpolate(conv(x), size=img_shape, mode='bilinear', align_corners=True)
            x = self.gn(self.prelu(x))
            late_sides.append(x)

        return late_sides

    def forward(self, img_shape, sides):
        late_sides = self.forward_sides(sides, img_shape)
        
        late_sides1 = torch.cat([late_sides[0],late_sides[1],late_sides[2],late_sides[3],late_sides[4],
                                 late_sides[5],late_sides[6],late_sides[7],late_sides[8],late_sides[9],
                                 late_sides[10]],1)

        fused = self.prelu(self.fused(late_sides1))
        late_sides.append(fused)

        return late_sides

class SEM(nn.Module):
    def __init__(self, ch_out, reduction=16):
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
            
        self.sem1 = SEM(ch_out, reduction=reduction)
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
    

class PFM(nn.Module):
    def __init__(self, ch_in, ch_out, ch_out_3x3e, pool_ch_out, EEM_ch_out, reduction, shortcut=False):
        super(PFM, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.03),
            nn.GroupNorm(4, ch_out)
            )
        ch_in1 = ch_out
        
        # 3x3 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(ch_in1, ch_out, kernel_size=3,padding=1,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, ch_out)
            )
        # 3x3 conv extended branch
        self.b2 = nn.Sequential(
            nn.Conv2d(ch_in1, ch_out_3x3e, kernel_size=3,padding=1,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out_3x3e, init=0.),
            nn.GroupNorm(4, ch_out_3x3e)
            )
        # 3x3 pool branch
        self.b3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_in1, pool_ch_out, kernel_size=1,padding=0,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, pool_ch_out)
            )
        if shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out+ch_out_3x3e, kernel_size=1,padding=0,groups=4,bias=False),
                nn.GroupNorm(4, ch_out+ch_out_3x3e)
                )
        
        self.EEM = EEM(ch_in1,EEM_ch_out,kernel=3,groups=ch_in1,reduction=reduction[0])
        self.sem1 = SEM(ch_out+ch_out_3x3e,reduction=reduction[1])
        self.sem2 = SEM(ch_out+ch_out_3x3e,reduction=reduction[1])
        self.prelu = nn.PReLU(num_parameters=ch_out+ch_out_3x3e, init=0.03)

    def forward(self, x, shortcut=False): 
        x1 = self.reducer(x)
        
        b1 = self.b1(x1) 
        b2 = self.b2(x1+b1)
        b3 = self.b3(x1)
        eem = self.EEM(x1)
        
        y1 = torch.cat([x1+b1+b3+eem,b2], 1)
        y2 = self.sem1(y1)
        
        if shortcut:
            x = self.shortcut(x)
        y3 = self.sem2(x)
        
        return self.prelu(x+y1+y2+y3)


class PDAM(nn.Module):
    def __init__(self, ch_in, ch_out, reduction, dropout):
        super(PDAM, self).__init__()
        self.conv1a = nn.Sequential(
            nn.Conv2d(ch_in[0],int(ch_out/2),kernel_size=1,padding=0,dilation=1,groups=1,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, int(ch_out/2))
            )
        self.conv1b = nn.Sequential(
            nn.Conv2d(ch_in[0],int(ch_out/2),kernel_size=1,padding=0,dilation=2,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, int(ch_out/2))
            )
        
        self.conv2a = nn.Sequential(
            nn.Conv2d(ch_in[1],int(ch_out/2),kernel_size=1,padding=0,dilation=1,groups=1,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        self.conv2b = nn.Sequential(
            nn.Conv2d(ch_in[1],int(ch_out/2),kernel_size=1,padding=0,dilation=2,groups=2,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        
        self.conv3a = nn.Sequential(
            nn.Conv2d(ch_in[2],int(ch_out/2),kernel_size=1,padding=0,dilation=1,groups=1,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        self.conv3b = nn.Sequential(
            nn.Conv2d(ch_in[2],int(ch_out/2),kernel_size=1,padding=0,dilation=2,groups=4,bias=False),
            nn.PReLU(num_parameters=int(ch_out/2), init=-0.01),
            nn.GroupNorm(4, int(ch_out/2))
            )
        
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_out,ch_out,kernel_size=1,padding=0,groups=2,bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(4, ch_out)
            )
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout2d(dropout)
        self.sem = SEM(ch_in[0], reduction=reduction)
        
    def forward(self,x,x1,x2):
        x0 = self.sem(x)
        x0 = torch.cat([self.conv1a(x+x0),self.conv1b(x+x0)],1)
        x1 = torch.cat([self.conv2a(x1),self.conv2b(x1)],1)
        x2 = torch.cat([self.conv3a(x2),self.conv3b(x2)],1)
        
        x3 = self.dropout(self.softmax(x1*x2))
        
        return self.conv4(x0+x1+x2+x3)
    

class LMM_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(LMM_Net, self).__init__()
        self.prelayer = nn.Conv2d(img_ch,96,kernel_size=3,padding=1,bias=False)
        self.sem1 = SEM(96, reduction=24)
        
        self.PFM1 = PFM(96, 32, 64, 32, 32, [8,24])
        self.PFM2 = PFM(96, 32, 64, 32, 32, [8,24])
        
        self.PFM3 = PFM(96, 32, 96, 32, 32, [8,32], True)
        self.PFM4 = PFM(128, 32, 96, 32, 32, [8,32])
        self.PFM5 = PFM(128, 32, 96, 32, 32, [8,32])
        
        self.PFM6 = PFM(128, 64, 128, 64, 64, [16,48], True)
        self.PFM7 = PFM(192, 64, 128, 64, 64, [16,48])
        self.PFM8 = PFM(192, 64, 128, 64, 64, [16,48])
        self.PFM9 = PFM(192, 64, 128, 64, 64, [16,48])
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3,padding=1,dilation=1,groups=4,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, 128)
            )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3,padding=2,dilation=2,groups=2,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, 128)
            )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3,padding=4,dilation=4,groups=4,bias=False),
            nn.ReLU(True),
            nn.GroupNorm(4, 64)
            )
        self.sem2 = SEM(320, reduction=80)
        
        self.PDAM4 = PDAM([320,192,192],128,64,0.0125)
        self.PDAM3 = PDAM([128,192,192],128,32,0.0125)
        self.PDAM2 = PDAM([128,128,128],96,32,0.025)
        self.PDAM1 = PDAM([96,96,96],64,12,0.05)
        
        self.FRCM = FRCM(ch_ins=[96,96,128,128,128,192,192,192,192,64,320],ch_out=2)
        
        self.sem3 = SEM(64+24, reduction=11)
        self.lastlayer = nn.Sequential(
            nn.Conv2d(64+24, 64, kernel_size=3,padding=1,groups=4,bias=False),
            nn.PReLU(num_parameters=64, init=-0.01),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 64, kernel_size=1,padding=0,bias=False),
            nn.PReLU(num_parameters=64, init=0.),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, output_ch,kernel_size=1,bias=False),
            nn.PReLU(num_parameters=output_ch, init=0.)
            )
        
        self.Max_Pooling = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.1)

    def forward(self,x):
        img_shape = x.shape[2:]
        x1 = self.prelayer(x)
        x2 = self.sem1(x1)
        x = x1+x2
        
        # encoding path
        i1 = self.PFM1(x)
        i2 = self.PFM2(i1)
        x = self.dropout1(self.Max_Pooling(i2))
        
        i3 = self.PFM3(x, True)
        i4 = self.PFM4(i3)
        i5 = self.PFM5(i4)
        x = self.dropout2(self.Max_Pooling(i5))

        i6 = self.PFM6(x, True)
        i7 = self.PFM7(i6)
        i8 = self.PFM8(i7)
        i9 = self.PFM9(i8)
        x = self.dropout3(self.Max_Pooling(i9))
        
        #D-Conv
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)
        x1 = torch.cat([x1,x2,x3],1)
        x2 = self.sem2(x1)
        
        # decoding path         
        x = F.interpolate(x1+x2, scale_factor=(2), mode='bilinear', align_corners=False)
        x = self.PDAM4(x,i9,i8)
        x = self.PDAM3(x,i7,i6)
        x = self.dropout3(x)
        
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
        x = self.PDAM2(x,i5,i4)
        x = self.dropout2(x)
        
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
        x = self.PDAM1(x,i2,i1)
        x = self.dropout1(x)
        
        sides = self.FRCM(img_shape,[i1,i2,i3,i4,i5,i6,i7,i8,i9,x,x1])
        x = torch.cat([x,sides[0],sides[1],sides[2],sides[3],sides[4],sides[5],
                       sides[6],sides[7],sides[8],sides[9],sides[10],sides[11]],1)
        
        x1 = self.sem3(x)

        return self.lastlayer(x+x1)
    
# MobileViT Block
# https://github.com/xmu-xiaoma666/External-Attention-pytorch?tab=readme-ov-file#28-MobileViT-Attention-Usage

# ULSAM
# https://github.com/Nandan91/ULSAM

# class EfficientCrackNet(nn.Module):
#     def __init__(self, img_ch=3, output_ch=1):
#         super(EfficientCrackNet, self).__init__()
#         self.DSC = 
#         self.EEM = EEM(ch_in1, EEM_ch_out, kernel=3, groups=ch_in1, reduction=reduction[0])
        
#     def forward(self, x):
#         pass 