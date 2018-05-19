'''
    Basic Components for GAN: Regressor,Segmentor, Discriminator
    Dong Nie
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnBuildUnits import *
from utils import weights_init


class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor,self).__init__()
        

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        #self.imsize = imsize

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = unetConvUnit(3, 64)
        self.conv_block64_128 = unetConvUnit(64, 128)
        self.conv_block128_256 = unetConvUnit(128, 256)
        self.conv_block256_512 = unetConvUnit(256, 512)
        #self.conv_block512_1024 = unetConvUnit(512, 1024)

        #self.up_block1024_512 = unetUpUnit(1024, 512)
        self.up_block512_256 = unetUpUnit(512, 256)
        self.up_block256_128 = unetUpUnit(256, 128)
        self.up_block128_64 = unetUpUnit(128, 64)

        self.last = nn.Conv2d(64, 4, 1)


    def forward(self, x):
        #print 'x.shape is, ',x.shape
        block1 = self.conv_block1_64(x)
#         print 'block1 size is ',block1.size()
        pool1 = self.pool1(block1)
#         print 'pool1 size is ',pool1.size()


        block2 = self.conv_block64_128(pool1)
#         print 'block2 size is ',block2.size()

        pool2 = self.pool2(block2)
#         print 'pool2 size is ',pool2.size()

        block3 = self.conv_block128_256(pool2)
#         print 'block3 size is ',block3.size()
        pool3 = self.pool3(block3)
#         print 'pool3 size is ',pool3.size()

        block4 = self.conv_block256_512(pool3)
#         print 'block4 size is ',block4.size()
#         pool4 = self.pool4(block4)1
#         print 'pool4 shape ',pool4.shape
#         block5 = self.conv_block512_1024(pool4)

#         up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)
#         print 'up2 size is ',up2.size()

        up3 = self.up_block256_128(up2, block2)
#         print 'up3 size is ',up3.size()

        up4 = self.up_block128_64(up3, block1)
#         print 'up4 size is ',up4.size()

#         return F.log_softmax(self.last(up4))
        return self.last(up4)

'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
We use more residual layers, especially one more layer after each residual layer which have a non-empty branch1
We use spatial dropout after each layer group
We use stochastic long-range connection
We use small dilation in the middle layers
'''
class DeeperResSegNet(nn.Module):
    def __init__(self, isRandomConnection = False, isSmallDilation = True, isSpatialDropOut = True, dropoutRate=0.25):
        super(DeeperResSegNet, self).__init__()
        #self.imsize = imsize
        
        self.isSpatialDropOut = isSpatialDropOut
        self.activation = F.relu
        
        
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #self.pool4 = nn.MaxPool2d(2)

        self.conv1_block1_64 = convUnit(3, 64)
        self.conv1_block64_64 = residualUnit3(64, 64, isEmptyBranch1=False)

        self.conv2_block64_128 = residualUnit3(64, 128)
        self.conv2_block128_128 = residualUnit3(128, 128, isEmptyBranch1=False)
        
        self.conv3_block128_256 = residualUnit3(128, 256)
        self.conv3_block256_256 = residualUnit3(256, 256, isEmptyBranch1=False)
        
        #dilated on the smallest resolution
        self.conv4_block256_512 = residualUnit3(256, 512, isDilation=isSmallDilation)
        
        self.dropout2d = nn.Dropout2d(dropoutRate)
         
        #self.conv_block512_1024 = unetConvUnit(512, 1024)

        #self.up_block1024_512 = unetUpUnit(1024, 512)
#         print 'line 107'
        if isRandomConnection:
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0.1)
    #         print 'line 109'
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0.05)
            
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0.01)
        else:
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0)
    #         print 'line 109'
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0)
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0)

        self.last = nn.Conv2d(64, 4, 1)
        
    def forward(self, x):   
#         print 'line 128: ','x.shape is, ',x.shape, 'type(x) is ',type(x)
        block0 = self.conv1_block1_64(x)
#         print 'block0 size is ', block0.size()
        block1 = self.conv1_block64_64(block0)
        pool1 = self.pool1(block1)
#         print 'pool1 size is ',pool1.size()


        block2 = self.conv2_block64_128(pool1)
        block2a = self.conv2_block128_128(block2)
        if self.isSpatialDropOut:
            block2a = self.dropout2d(block2a)
#         print 'block2 size is ',block2.size()

        pool2 = self.pool2(block2a)
#         print 'pool2 size is ',pool2.size()

        block3 = self.conv3_block128_256(pool2)
        block3a = self.conv3_block256_256(block3)
        if self.isSpatialDropOut:
            block3a = self.dropout2d(block3a)
        
#         print 'block3 size is ',block3.size()
        pool3 = self.pool3(block3a)
#         print 'pool3 size is ',pool3.size()

        block4 = self.conv4_block256_512(pool3)
        #actually, I donot think this is good for this layer, if we do dropout here, then block4 and block3 are all randomized
        if self.isSpatialDropOut: 
            block4 = self.dropout2d(block4)
            
#         print 'block4 size is ',block4.size()
#         pool4 = self.pool4(block4)1
#         print 'pool4 shape ',pool4.shape
#         block5 = self.conv_block512_1024(pool4)

#         up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)

#         print 'up2 size is ',up2.size()

        up3 = self.up_block256_128(up2, block2)
#         print 'up3 size is ',up3.size()

        up4 = self.up_block128_64(up3, block1)
#         print 'up4 size is ',up4.size()

#         return F.log_softmax(self.last(up4))
        return self.last(up4)

'''
Dilated network:
a segmentation network introduced in "Adversarial training and dilated convolutions for brain MRI segmentation"
from url: https://arxiv.org/pdf/1707.03195.pdf
'''
class DilatedNetwork(nn.Module):
    def __init__(self):
        super(DilatedNetwork, self).__init__()
        #self.imsize = imsize

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = unetConvUnit(3, 64)
        self.conv_block64_128 = unetConvUnit(64, 128)
        self.conv_block128_256 = unetConvUnit(128, 256)
        self.conv_block256_512 = unetConvUnit(256, 512)
        #self.conv_block512_1024 = unetConvUnit(512, 1024)

        #self.up_block1024_512 = unetUpUnit(1024, 512)
        self.up_block512_256 = unetUpUnit(512, 256)
        self.up_block256_128 = unetUpUnit(256, 128)
        self.up_block128_64 = unetUpUnit(128, 64)

        self.last = nn.Conv2d(64, 4, 1)


    def forward(self, x):
        #print 'x.shape is, ',x.shape
        block1 = self.conv_block1_64(x)
#         print 'block1 size is ',block1.size()
        pool1 = self.pool1(block1)
#         print 'pool1 size is ',pool1.size()


        block2 = self.conv_block64_128(pool1)
#         print 'block2 size is ',block2.size()

        pool2 = self.pool2(block2)
#         print 'pool2 size is ',pool2.size()

        block3 = self.conv_block128_256(pool2)
#         print 'block3 size is ',block3.size()
        pool3 = self.pool3(block3)
#         print 'pool3 size is ',pool3.size()

        block4 = self.conv_block256_512(pool3)
#         print 'block4 size is ',block4.size()
#         pool4 = self.pool4(block4)1
#         print 'pool4 shape ',pool4.shape
#         block5 = self.conv_block512_1024(pool4)

#         up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)
#         print 'up2 size is ',up2.size()

        up3 = self.up_block256_128(up2, block2)
#         print 'up3 size is ',up3.size()

        up4 = self.up_block128_64(up3, block1)
#         print 'up4 size is ',up4.size()

#         return F.log_softmax(self.last(up4))
        return self.last(up4)

'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
We use spatial dropout after each layer group
We use stochastic long-range connection (short-range connection can be directly set in the residual module)
We use small dilation in the middle layers
We use local conv in the network
'''
class ResSegNet_v1(nn.Module):
    def __init__(self, in_channel = 3, n_classes = 4, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None, dropoutRate=0.25):
        super(ResSegNet_v1, self).__init__()
        #self.imsize = imsize
    
        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation
        
        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = convUnit(in_channel, 64)
        self.conv_block64_64 = residualUnit3(64, 64, isDilation=False,isEmptyBranch1=False, spatial_dropout_rate=0.01)

        self.conv_block64_128 = residualUnit3(64, 128, isDilation=False,isEmptyBranch1=False,spatial_dropout_rate=0.01)
        
        self.conv_block128_256 = residualUnit3(128, 256, isDilation=False,isEmptyBranch1=False,spatial_dropout_rate=0.01)
        
        #the residual layers on the smallest resolution
        self.conv_block256_512 = residualUnit3(256, 512, isDilation=isSmallDilation, isEmptyBranch1=False)
        #self.conv_block512_1024 = unetConvUnit(512, 1024)

        #self.up_block1024_512 = unetUpUnit(1024, 512)
#         print 'line 107'

        self.dropout2d = nn.Dropout2d(dropoutRate)
        
#         self.up_block512_256 = ResUpUnit(512, 256)
# #         print 'line 109'
#         self.up_block256_128 = ResUpUnit(256, 128)
#         self.up_block128_64 = ResUpUnit(128, 64)
        
        if isRandomConnection:
#             print 'line 233: isRandomConnection'
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0.1)
        #         print 'line 109'
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0.1)
        
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0.1)
        else:
#             print 'line 240: isRandomConnection'
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0)
        #         print 'line 109'
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0)
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0)

        self.localconv = nn.Conv2dLocal(64, 64, 168, 112, kernel_size=3, stride=1, padding=1)
        self.last = nn.Conv2d(64, n_classes, 1)
        
        
    def forward(self, x):   
        #print 'x.shape is, ',x.shape
        block0 = self.conv_block1_64(x)
#         print 'block0 size is ', block0.size()
        block1 = self.conv_block64_64(block0)
        pool1 = self.pool1(block1)
#         print 'pool1 size is ',pool1.size()


        block2 = self.conv_block64_128(pool1)
        if self.isSpatialDropOut:
            block2 = self.dropout2d(block2)
#         print 'block2 size is ',block2.size()

        pool2 = self.pool2(block2)
#         print 'pool2 size is ',pool2.size()

        block3 = self.conv_block128_256(pool2)
        if self.isSpatialDropOut:
            block3 = self.dropout2d(block3)
#             print 'line 264: spatial dropout done'
        
#         if self.isRandomConnection:
#             print 'line 201: random connection'
#         if self.isSmallDilation:
#             print 'line 203: small dilation'
#         if self.isSpatialDropOut:
#             print 'line 205: spatial dropout'
        
        
#         print 'block3 size is ',block3.size()
        pool3 = self.pool3(block3)
#         print 'pool3 size is ',pool3.size()

        block4 = self.conv_block256_512(pool3)
#         if self.isSpatialDropOut:
#             block4 = self.dropout2d(block4)
#         print 'block4 size is ',block4.size()
#         pool4 = self.pool4(block4)1
#         print 'pool4 shape ',pool4.shape
#         block5 = self.conv_block512_1024(pool4)

#         up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)
#         print 'up2 size is ',up2.size()

        up3 = self.up_block256_128(up2, block2)
#         print 'up3 size is ',up3.size()

        up4 = self.up_block128_64(up3, block1)
#         print 'up4 size is ',up4.size()

#         return F.log_softmax(self.last(up4))
        localconv = self.localconv(up4)
        return self.last(localconv)


'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
We use spatial dropout after each layer group
We use stochastic long-range connection (short-range connection can be directly set in the residual module)
We use small dilation in the middle layers
'''
class ResSegNet(nn.Module):
    def __init__(self, in_channel = 3, n_classes = 4, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None, dropoutRate=0.25):
        super(ResSegNet, self).__init__()
        #self.imsize = imsize
    
        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation
        
        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = convUnit(in_channel, 64)
        self.conv_block64_64 = residualUnit3(64, 64, isDilation=False,isEmptyBranch1=False, spatial_dropout_rate=0.01)

        self.conv_block64_128 = residualUnit3(64, 128, isDilation=False,isEmptyBranch1=False,spatial_dropout_rate=0.01)
        
        self.conv_block128_256 = residualUnit3(128, 256, isDilation=False,isEmptyBranch1=False,spatial_dropout_rate=0.01)
        
        #the residual layers on the smallest resolution
        self.conv_block256_512 = residualUnit3(256, 512, isDilation=isSmallDilation, isEmptyBranch1=False)
        #self.conv_block512_1024 = unetConvUnit(512, 1024)

        #self.up_block1024_512 = unetUpUnit(1024, 512)
#         print 'line 107'

        self.dropout2d = nn.Dropout2d(dropoutRate)
        
#         self.up_block512_256 = ResUpUnit(512, 256)
# #         print 'line 109'
#         self.up_block256_128 = ResUpUnit(256, 128)
#         self.up_block128_64 = ResUpUnit(128, 64)
        
        if isRandomConnection:
#             print 'line 233: isRandomConnection'
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0.1)
        #         print 'line 109'
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0.1)
        
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0.1)
        else:
#             print 'line 240: isRandomConnection'
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0)
        #         print 'line 109'
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0)
            self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0)

        self.last = nn.Conv2d(64, n_classes, 1)
        
        
    def forward(self, x):   
        #print 'x.shape is, ',x.shape
        block0 = self.conv_block1_64(x)
#         print 'block0 size is ', block0.size()
        block1 = self.conv_block64_64(block0)
        pool1 = self.pool1(block1)
#         print 'pool1 size is ',pool1.size()


        block2 = self.conv_block64_128(pool1)
        if self.isSpatialDropOut:
            block2 = self.dropout2d(block2)
#         print 'block2 size is ',block2.size()

        pool2 = self.pool2(block2)
#         print 'pool2 size is ',pool2.size()

        block3 = self.conv_block128_256(pool2)
        if self.isSpatialDropOut:
            block3 = self.dropout2d(block3)
#             print 'line 264: spatial dropout done'
        
#         if self.isRandomConnection:
#             print 'line 201: random connection'
#         if self.isSmallDilation:
#             print 'line 203: small dilation'
#         if self.isSpatialDropOut:
#             print 'line 205: spatial dropout'
        
        
#         print 'block3 size is ',block3.size()
        pool3 = self.pool3(block3)
#         print 'pool3 size is ',pool3.size()

        block4 = self.conv_block256_512(pool3)
#         if self.isSpatialDropOut:
#             block4 = self.dropout2d(block4)
#         print 'block4 size is ',block4.size()
#         pool4 = self.pool4(block4)1
#         print 'pool4 shape ',pool4.shape
#         block5 = self.conv_block512_1024(pool4)

#         up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)
#         print 'up2 size is ',up2.size()

        up3 = self.up_block256_128(up2, block2)
#         print 'up3 size is ',up3.size()

        up4 = self.up_block128_64(up3, block1)
#         print 'up4 size is ',up4.size()

#         return F.log_softmax(self.last(up4))

        return self.last(up4)


'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
Besides the segmentation maps, we also return the contours
We use spatial dropout after each layer group
We use stochastic long-range connection
We use small dilation in the middle layers
'''
class ResSegContourNet(nn.Module):
    def __init__(self, in_channel = 3, n_classes = 4, isRandomConnection=None, isSmallDilation=None, isSpatialDropOut=None, dropoutRate=0.25):
        super(ResSegContourNet, self).__init__()
        #self.imsize = imsize
    
        self.isSpatialDropOut = isSpatialDropOut
        self.isRandomConnection = isRandomConnection
        self.isSmallDilation = isSmallDilation
        
        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = convUnit(in_channel, 64)
        self.conv_block64_64 = residualUnit3(64, 64, isDilation=False,isEmptyBranch1=False)

        self.conv_block64_128 = residualUnit3(64, 128, isDilation=False,isEmptyBranch1=False)
        
        self.conv_block128_256 = residualUnit3(128, 256, isDilation=False,isEmptyBranch1=False)
        
        #the residual layers on the smallest resolution
        self.conv_block256_512 = residualUnit3(256, 512, isDilation=isSmallDilation, isEmptyBranch1=False)
        #self.conv_block512_1024 = unetConvUnit(512, 1024)

        #self.up_block1024_512 = unetUpUnit(1024, 512)
#         print 'line 107'

        self.dropout2d = nn.Dropout2d(dropoutRate)
        
#         self.up_block512_256 = ResUpUnit(512, 256)
# #         print 'line 109'
#         self.up_block256_128 = ResUpUnit(256, 128)
#         self.up_block128_64 = ResUpUnit(128, 64)
        
        if isRandomConnection:
#             print 'line 233: isRandomConnection'
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0.1)
        #         print 'line 109'
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0.05)
        
#             self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0.01)
        else:
#             print 'line 240: isRandomConnection'
            self.up_block512_256 = ResUpUnit(512, 256, spatial_dropout_rate = 0)
        #         print 'line 109'
            self.up_block256_128 = ResUpUnit(256, 128, spatial_dropout_rate = 0)
#             self.up_block128_64 = ResUpUnit(128, 64, spatial_dropout_rate = 0)


        self.up_block128_64 = BaseResUpUnit(128, 64)
        
        #### for segmentation
        self.seg_conv1 = residualUnit3(64,64)
        self.seg_conv2 = residualUnit3(64,32)
        self.seg_last = nn.Conv2d(32, n_classes, 1)
        
        ##### for contour classification
        self.contour_conv1 = residualUnit3(64, 64)
        self.contour_conv2 = residualUnit3(64, 32)
        self.contour_last = nn.Conv2d(32, 2, 1)
        
#         self.last = nn.Conv2d(64, n_classes, 1)
        
        
    def forward(self, x):   
        #print 'x.shape is, ',x.shape
        block0 = self.conv_block1_64(x)
#         print 'block0 size is ', block0.size()
        block1 = self.conv_block64_64(block0)
        pool1 = self.pool1(block1)
#         print 'pool1 size is ',pool1.size()


        block2 = self.conv_block64_128(pool1)
        if self.isSpatialDropOut:
            block2 = self.dropout2d(block2)
#         print 'block2 size is ',block2.size()

        pool2 = self.pool2(block2)
#         print 'pool2 size is ',pool2.size()

        block3 = self.conv_block128_256(pool2)
        if self.isSpatialDropOut:
            block3 = self.dropout2d(block3)
#             print 'line 264: spatial dropout done'
        
#         if self.isRandomConnection:
#             print 'line 201: random connection'
#         if self.isSmallDilation:
#             print 'line 203: small dilation'
#         if self.isSpatialDropOut:
#             print 'line 205: spatial dropout'
        
        
#         print 'block3 size is ',block3.size()
        pool3 = self.pool3(block3)
#         print 'pool3 size is ',pool3.size()

        block4 = self.conv_block256_512(pool3)
#         if self.isSpatialDropOut:
#             block4 = self.dropout2d(block4)
#         print 'block4 size is ',block4.size()
#         pool4 = self.pool4(block4)1
#         print 'pool4 shape ',pool4.shape
#         block5 = self.conv_block512_1024(pool4)

#         up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)
#         print 'up2 size is ',up2.size()

        up3 = self.up_block256_128(up2, block2)
#         print 'up3 size is ',up3.size()

        up4 = self.up_block128_64(up3, block1)
#         print 'up4 size is ',up4.size()

#         return F.log_softmax(self.last(up4))

        # seg conv1
        seg_conv1 = self.seg_conv1(up4)
        seg_conv2 = self.seg_conv2(seg_conv1)
        
        # regression task1
        contour_conv1 = self.contour_conv1(up4)
        contour_conv2 = self.contour_conv2(contour_conv1)
        
        #         return F.log_softmax(self.last(up4))
        return self.seg_last(seg_conv2), self.contour_last(contour_conv2)
    
    
'''
Residual Learning for FCN: short-range residual connection and long-range residual connection
Multi-task Learning: Using segmentation and regression for the same network
'''
class ResSegRegNet(nn.Module):
    def __init__(self):
        super(ResSegNet, self).__init__()
        #self.imsize = imsize

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = convUnit(3, 64)
        self.conv_block64_64 = residualUnit3(64, 64)

        self.conv_block64_128 = residualUnit3(64, 128)
        self.conv_block128_256 = residualUnit3(128, 256)
        self.conv_block256_512 = residualUnit3(256, 512)
        #self.conv_block512_1024 = unetConvUnit(512, 1024)

        #self.up_block1024_512 = unetUpUnit(1024, 512)
#         print 'line 107'
        self.up_block512_256 = ResUpUnit(512, 256)
#         print 'line 109'
        self.up_block256_128 = ResUpUnit(256, 128)
        self.up_block128_64 = BaseResUpUnit(128, 64)
        
        #### for segmentation
        self.seg_conv1 = residualUnit3(64,64)
        self.seg_conv2 = residualUnit3(64,32)
        self.seg_last = nn.Conv2d(32, 4, 1)
        
        ##### for regression 1
        self.reg1_conv1 = residualUnit3(64, 64)
        self.reg1_conv2 = residualUnit3(64, 32)
        self.reg1_last = nn.Conv2d(32, 1, 1)
        
        ##### for regression 2
        self.reg2_conv1 = residualUnit3(64, 64)
        self.reg2_conv2 = residualUnit3(64, 32)
        self.reg2_last = nn.Conv2d(32, 1, 1)        
 
        ##### for regression 3
        self.reg3_conv1 = residualUnit3(64, 64)
        self.reg3_conv2 = residualUnit3(64, 32)
        self.reg3_last = nn.Conv2d(32, 1, 1)   
        
        #....          
        
    def forward(self, x):   
        #print 'x.shape is, ',x.shape
        block0 = self.conv_block1_64(x)
#         print 'block0 size is ', block0.size()
        block1 = self.conv_block64_64(block0)
        pool1 = self.pool1(block1)
#         print 'pool1 size is ',pool1.size()


        block2 = self.conv_block64_128(pool1)
#         print 'block2 size is ',block2.size()

        pool2 = self.pool2(block2)
#         print 'pool2 size is ',pool2.size()

        block3 = self.conv_block128_256(pool2)
#         print 'block3 size is ',block3.size()
        pool3 = self.pool3(block3)
#         print 'pool3 size is ',pool3.size()

        block4 = self.conv_block256_512(pool3)
#         print 'block4 size is ',block4.size()
#         pool4 = self.pool4(block4)1
#         print 'pool4 shape ',pool4.shape
#         block5 = self.conv_block512_1024(pool4)

#         up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)
#         print 'up2 size is ',up2.size()

        up3 = self.up_block256_128(up2, block2)
#         print 'up3 size is ',up3.size()

        up4 = self.up_block128_64(up3, block1)
        
        # seg conv1
        seg_conv1 = self.seg_conv1(up4)
        seg_conv2 = self.seg_conv2(seg_conv1)
        
        # regression task1
        reg1_conv1 = self.reg1_conv1(up4)
        reg1_conv2 = self.reg1_conv2(reg1_conv1)
        
        # regression task2
        reg2_conv1 = self.reg2_conv1(up4)
        reg2_conv2 = self.reg2_conv2(reg2_conv1)
        
        # regression task3
        reg3_conv1 = self.reg3_conv1(up4)
        reg3_conv2 = self.reg3_conv2(reg3_conv1)
        
#         print 'up4 size is ',up4.size()

#         return F.log_softmax(self.last(up4))
        return self.seg_last(seg_conv2), self.reg1_last(reg1_conv2), self.reg2_last(reg2_conv2), self.reg3_last(reg3_conv2)
    
class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample = upsample

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=1)

        for i in range(self.n_residual_blocks):
            self.add_module('res' + str(i+1), residualUnit(64))

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        in_channels = 64
        out_channels = 256
        for i in range(self.upsample):
            self.add_module('upscale' + str(i+1), upsampleUnit(in_channels, out_channels))
            in_channels = out_channels
            out_channels = out_channels/2

        self.conv3 = nn.Conv2d(in_channels, 3, 9, stride=1, padding=1)

    def forward(self, x):
        x = F.elu(self.conv1(x))

        y = self.__getattr__('res1')(x)
        for i in range(1, self.n_residual_blocks):
            y = self.__getattr__('res' + str(i+1))(y)

        x = self.conv2(y) + x

        for i in range(self.upsample):
            x = self.__getattr__('upscale' + str(i+1))(x)

        return F.sigmoid(self.conv3(x))

class regGenerator(nn.Module):
    def __init__(self):
        super(regGenerator, self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.residual = self.make_layer(residualUnit3, 15)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        self.conv6 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.conv8 = nn.Conv2d(128, 128, 3, stride=2, padding=1)

        self.fc1 = nn.Linear(9856, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
#         print 'line 260 ',x.size()
        x = F.elu(self.conv2(x))
#         print 'line 262 ',x.size()
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
#         print 'line 265 ',x.size()
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
#         print 'line 267 ',x.size()
        x = F.elu(self.conv7(x))
        x = F.elu(self.conv8(x))
#         print 'line 271 ',x.size()

        # Flatten
        x = x.view(x.size(0), -1)
#         print 'line 266 is ', x.size()

        x = F.elu(self.fc1(x))
        return F.sigmoid(self.fc2(x))