
'''
    * Building units for neural networks: residual units, unet units, upsampling unit and so on.
    * Dong Nie
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function
from itertools import repeat



'''
ordinary conv block
'''
class convUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu):
        super(convUnit, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride, padding)
        init.xavier_uniform(self.conv.weight, gain = np.sqrt(2.0))
        init.constant(self.conv.bias, 0)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
'''    
 two-layer residual unit: two conv without BN and identity mapping
'''
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride, padding)
        init.xavier_uniform(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv1.bias, 0)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride, padding)
        init.xavier_uniform(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.conv2.bias, 0)

    def forward(self, x):
        return F.relu(self.conv2(F.elu(self.conv1(x))) + x)
'''
    two-layer residual unit: two conv with BN and identity mapping
'''    
class residualUnit1(nn.Module): 
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu):
        super(residualUnit1, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride, padding, bias=False)
        init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2.0)) #or gain=1
        init.constant(self.conv1.bias, 0)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride, padding, bias=False)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0)) #or gain=1
        init.constant(self.conv2.bias, 0)
        self.bn2 = nn.BatchNorm2d(out_size)
        
    def forward(self, x): 
        identity_data = x
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = torch.add(output,identity_data)
        output = self.relu(output)
        return output 

'''
three-layer residual unit: three conv with BN and identity mapping
this one doesn't change the size of channels, which means the in_size is same with out_size

input: 
    x
output:
    bottleneck residual block
By Dong Nie
'''
class residualUnit3(nn.Module): 
    def __init__(self, in_size, out_size, isDilation=None, isEmptyBranch1=None, spatial_dropout_rate=0, activation=F.relu):
        super(residualUnit3, self).__init__()
        #         mid_size = in_size/2
        mid_size = out_size/2 ###I think it should better be half the out size instead of the input size
#         print 'line 74, in and out size are, ',in_size,' ',mid_size

        if isDilation:
            self.conv1 = nn.Conv2d(in_channels=in_size, out_channels=mid_size, kernel_size=1, stride=1, padding=0, dilation=2,  bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_size, out_channels=mid_size, kernel_size=1, stride=1, padding=0, bias=False)
        init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2.0)) #or gain=1
#         init.constant(self.conv1.bias, 0)
        self.bn1 = nn.BatchNorm2d(mid_size)
        self.relu = nn.ReLU()
        
        if isDilation:
            self.conv2 = nn.Conv2d(in_channels=mid_size, out_channels=mid_size, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(in_channels=mid_size, out_channels=mid_size, kernel_size=3, stride=1, padding=1, bias=False)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0)) #or gain=1
#         init.constant(self.conv2.bias, 0)
        self.bn2 = nn.BatchNorm2d(mid_size)
        
        if isDilation:
            self.conv3 = nn.Conv2d(in_channels=mid_size, out_channels=out_size, kernel_size=1, stride=1, padding=0, dilation=2, bias=False)
        else:
            self.conv3 = nn.Conv2d(in_channels=mid_size, out_channels=out_size, kernel_size=1, stride=1, padding=0, bias=False)
        
        init.xavier_uniform(self.conv3.weight, gain=np.sqrt(2.0)) #or gain=1
        self.dropout2d = nn.Dropout2d(spatial_dropout_rate)
        self.spatial_dropout_rate = spatial_dropout_rate
#         init.constant(self.conv3.bias, 0)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.isEmptyBranch1 = isEmptyBranch1
        if in_size!=out_size or isEmptyBranch1==False:
            if isDilation:
                self.convX = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1, padding=0, dilation=2, bias=False)
            else:
                self.convX = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1, padding=0, bias=False)
            self.bnX = nn.BatchNorm2d(out_size)
        
    def forward(self, x): 
        identity_data = x
#         print 'line 94, size of x is ', x.size()
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        outSZ = output.size()
        idSZ = identity_data.size()
        if outSZ[1]!=idSZ[1] or self.isEmptyBranch1==False:
            identity_data = self.bnX(self.convX(identity_data))
#         print output.size(), identity_data.size()
        if self.spatial_dropout_rate>0:
            identity_data = self.dropout2d(identity_data)
        output = torch.add(output,identity_data)
        output = self.relu(output)
        return output 

    
'''
 long-term residual unit, there is a long way (a lot of convs) before the residual addition
 
 By Dong Nie
'''
class longResidualUnit(nn.Module): 
    def __init__(self,in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu):
        super(residualUnit1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2.0)) #or gain=1
        init.constant(self.conv1.bias, 0)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x): 
        identity_data = x
        output = self.relu(self.bn1(self.conv1(x)))
        output = torch.add(output,identity_data)
        output = self.relu(output)
        return output 


'''
    Residual upsampling block with long-range residual connection
input:
    x: the current layer you want to consider
    bridge: the one you want to combine (using summary instead of concatenation) from lower layers
output:
    residual output
    space_dropout_rate: the rate to make several of the feature maps to be zero (to avoid the correlation within a feature map, we use 
    spatial dropout instead of traditional dropout
    By Dong Nie
'''
class ResUpUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, spatial_dropout_rate=0, isConvDilation=None):
        super(ResUpUnit, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=True)
        init.xavier_uniform(self.up.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant(self.up.bias, 0)
#         self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=True)
#         init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0)) #or gain=1
#         init.constant(self.conv.bias, 0)
        self.conv = residualUnit3(out_size, out_size, isDilation=isConvDilation)

#         self.SpatialDroput = nn.SpatialDropout(space_dropout_rate)
        self.dropout2d = nn.Dropout2d(spatial_dropout_rate)
        self.spatial_dropout_rate = spatial_dropout_rate
        self.conv2 = residualUnit3(out_size, out_size, isDilation=isConvDilation, isEmptyBranch1=False)
#         print 'line 147, in_size is ',out_size,' out_size is ',out_size

        self.relu = nn.ReLU()

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):#bridge is the corresponding lower layer
        up = self.up(x)
#         print 'line 158 ',up.size()
        crop1 = self.center_crop(bridge, up.size()[2])
#         crop1_dp = self.SpatialDroput(crop1)
        if self.spatial_dropout_rate>0:
            crop1 = self.dropout2d(crop1)
        out = self.relu(torch.add(up, crop1))
#         print 'line 161'
        a = self.conv(out)
#         print 'line 161 is ', a.size()
        out = self.relu(a)
        out = self.relu(self.conv2(out))
#         print 'line 163 is ', out.size()
        return out


'''
    Basic Residual upsampling block with long-range residual connection. Note we didn't have two 
    short residual blocks after the long-range residual operation. 
input:
    x: the current layer you want to consider
    bridge: the one you want to combine (using summary instead of concatenation) from lower layers
output:
    residual output
    
    By Dong Nie
'''
class BaseResUpUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(BaseResUpUnit, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=True)
        init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0)) #or gain=1
        init.constant(self.up.bias, 0)
#         self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=True)
#         init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0)) #or gain=1
#         init.constant(self.conv.bias, 0)
#         self.conv = residualUnit3(out_size, out_size)
#         print 'line 142, in_size is ',in_size,' out_size is ',out_size
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=True)
#         init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0)) #or gain=1
#         init.constant(self.conv2.bias, 0)
#         self.conv2 = residualUnit3(out_size, out_size)
#         print 'line 147, in_size is ',out_size,' out_size is ',out_size

        self.relu = nn.ReLU()

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):#bridge is the corresponding lower layer
        up = self.up(x)
#         print 'line 158 ',up.size()
        crop1 = self.center_crop(bridge, up.size()[2])
        out = self.relu(torch.add(up, crop1))
#         print 'line 161'
#         a = self.conv(out)
# #         print 'line 161 is ', a.size()
#         out = self.relu(a)
#         out = self.relu(self.conv2(out))
#         print 'line 163 is ', out.size()
        return out
    
class upsampleUnit(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleUnit, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        return F.elu(self.conv1(self.upsample1(x)))

class unetConvUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(unetConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=True)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0)) #or gain=1
        init.constant(self.conv.bias, 0)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=True)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0)) #or gain=1
        init.constant(self.conv2.bias, 0)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))

        return out

'''
    unet upsampling block
'''
class unetUpUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(unetUpUnit, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=True)
        init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0)) #or gain=1
        init.constant(self.up.bias, 0)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=True)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0)) #or gain=1
        init.constant(self.conv.bias, 0)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=True)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0)) #or gain=1
        init.constant(self.conv2.bias, 0)
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):#bridge is the corresponding lower layer
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))

        return out
 
'''
    The cross entropy loss for 3D data, usually used in voxel wise segmentation.
input:
    predict: 5D tensor, even Variable
    target: 4D tensor, even Variable
output:
    loss
    
    By Dong Nie
'''
class CrossEntropy3d(nn.Module):

    def __init__(self, weight = None, size_average=True, ignore_label=255):
        '''weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"'''
        super(CrossEntropy3d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w, d)
                target:(n, h, w, d): 0,1,...,C-1       
        """
        assert not target.requires_grad
        assert predict.dim() == 5
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
        assert predict.size(4) == target.size(3), "{0} vs {1} ".format(predict.size(4), target.size(3))
        n, c, h, w, d = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label) #actually, it doesn't convert to one-hot format
        target = target[target_mask] #N*1
        predict = predict.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous() # n, h, w, d, c
        predict = predict[target_mask.view(n, h, w, d, 1).repeat(1, 1, 1, 1, c)].view(-1, c) #N*C
        loss = F.cross_entropy(predict, target, weight = self.weight, size_average = self.size_average)
#         print 'cross-entropy-loss: ',type(loss)
        return loss


'''
    The cross entropy loss for 2D data, usually used in pixel wise segmentation.
input:
    predict: 4D tensor, even Variable
    target: 3D tensor, even Variable
output:
    loss
    
    By Dong Nie
'''
class CrossEntropy2d(nn.Module):

    def __init__(self, weight = None, size_average=True, ignore_label=255):
        '''weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"'''
        super(CrossEntropy2d, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w): 0,1,...,C-1       
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight = self.weight, size_average = self.size_average)
#         print 'cross-entropy-loss: ',type(loss)
        return loss



'''
    The cross entropy loss for 2D dataset (usually used in FCN based segmentation)
    implemented by using nn.LLLoss2d, we use F.log_softmax to implement log_softmax in 2D.
    This one is believed to be stable and also faster
'''    
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
     
 
    
'''
    This criterion is a implementation of Focal Loss, which is proposed in 
    Focal Loss for Dense Object Detection.
    
    Now I implement it in the environment of segmentation

        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.

Input:
    class_num: the number of categories
    alpha(1D Tensor, Variable) : the scalar factor for this criterion
    gamma: aim at reducing the relative loss for the well classified examples, focusing more on hard, misclassified example
    size_average: true by default, the losses are averaged over observations for each minibatch, if set by false, it will summaried for each minibatch
    inputs: a 4D tensor for the predicted segmentation maps (before softmax), NXCXWXH
    targets: a 3D tensor for the ground truth segmentation maps, NXWXH
Output:
    The averaged losses

10/31/2017
By Dong Nie
'''
class myFocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(myFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        assert inputs.dim()==4,'inputs size should be 4: NXCXWXH'
        N = inputs.size(0)
        C = inputs.size(1)
        W = inputs.size(2)
        H = inputs.size(3)
        P = F.softmax(inputs)

        ## one hot embeding for the targets ##
        class_mask = inputs.data.new(N, C, W, H).fill_(0)
        class_mask = Variable(class_mask)
        
        targets = torch.unsqueeze(targets,1) #Nx1xHxW
        class_mask.scatter_(1, targets, 1) #scatter along the 'numOfDims' dimension
        
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.) 
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
        alpha = 0.25

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


'''
    Dice cost function for a single organ
input:
    input: a torch variable of size BatchxnclassesxHxW representing probabilities for each class
    target: a a also tensor, with batchx1xHxW
output:
    Variable scalar loss
    succeed for two type segmentation
'''
def myDiceLoss4Organ(input,target):
#     assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
    one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
    two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))

    target1 = Variable(torch.unsqueeze(target.data,1)) #Nx1xHxW
    target_one_hot = Variable(torch.cuda.FloatTensor(input.size()).zero_()) #NxCxHxW
#     target_one_hot = target_one_hot.permute(0,2,3,1) #NxHxWxC
    target_one_hot.scatter_(1, target1, 1) #scatter along the 'numOfDims' dimension
    
    uniques=np.unique(target_one_hot.data.cpu().numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

#     print 'line 330: size: ',target_one_hot.size()
    
    probs = F.softmax(input) #maybe it is not necessary
#     print 'line 331: size: ',probs.size()
    
    target = target_one_hot.contiguous().view(-1,1).squeeze(1)
    result = probs.contiguous().view(-1,1).squeeze(1)
    #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
    
#     intersect = torch.dot(result, target) #it doesn't support autograd
    
    intersect_vec = result * target
    intersect = torch.sum(intersect_vec)
        
    target_sum = torch.sum(target)
    result_sum = torch.sum(result)
    union = result_sum + target_sum + (two*eps)
#     print 'type of union: ',type(union)

    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#             out = torch.add(out, IoU.data*2)
    dice_total = one - two*IoU

#     dice_total = -1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
#     print 'type of dice_total: ', type(dice_total)
    return dice_total


'''
    This is generalized dice loss for multiple organs, which means you can compute dice for more than one organ at a time
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
    
output:
    loss: scalar
    
    April, 2018, let's test, succeed, suggest use this one
'''
   
class GeneralizedDiceLoss4Organs(nn.Module):
    def __init__(self,  organIDs = [1], size_average=True):   
        super(GeneralizedDiceLoss4Organs,self).__init__()
        self.organIDs = organIDs
        self.size_average = size_average
        
    def forward(self, inputs, targets, save=True):
        """
            Args:
                inputs:(n, c, h, w, d)
                targets:(n, h, w, d): 0,1,...,C-1       
        """
        assert not targets.requires_grad
        assert inputs.dim() == 5, inputs.shape
        assert targets.dim() == 4, targets.shape
        assert inputs.size(0) == targets.size(0), "{0} vs {1} ".format(inputs.size(0), targets.size(0))
        assert inputs.size(2) == targets.size(1), "{0} vs {1} ".format(inputs.size(2), targets.size(1))
        assert inputs.size(3) == targets.size(2), "{0} vs {1} ".format(inputs.size(3), targets.size(2))
        assert inputs.size(4) == targets.size(3), "{0} vs {1} ".format(inputs.size(4), targets.size(3))
        
        
        eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
        one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
        two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))
                          
        inputSZ = inputs.size() #it should be sth like NxCxHxW
        
        inputs = F.softmax(inputs, dim=1)
    
        numOfCategories = inputSZ[1]
        assert numOfCategories==len(self.organIDs), 'organ weights is not matched with organs (bg should be included)'
        ####### Convert categorical to one-hot format

        results_one_hot = inputs
        
        
        target1 = Variable(torch.unsqueeze(targets.data,1)) #Nx1xHxW
        targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_()) #NxCxHxW
        targets_one_hot.scatter_(1, target1, 1) #scatter along the 'numOfDims' dimension
 
        ###### Now the prediction and target has become one-hot format
        ###### Compute the dice for each organ
        out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad = True)
    #     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
    #     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)

        intersect = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
        union = Variable(torch.cuda.FloatTensor(1).fill_(0.0))
        
        for organID in range(0, numOfCategories):

            target = targets_one_hot[:,organID,...].contiguous().view(-1,1).squeeze(1) #for 2D or 3D
            result = results_one_hot[:,organID,...].contiguous().view(-1,1).squeeze(1) #for 2D or 3D
    #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
            if torch.sum(target)==0:
                organWeight =0
            else:
                organWeight = 1/((torch.sum(target))**2+eps)
#             print 'sum: %d'%torch.sum(target),' organWeight: %f'%organWeight
    #         intersect = torch.dot(result, target)
            intersect_vec = result * target
            intersect = intersect + organWeight*torch.sum(intersect_vec)
    #         print type(intersect)
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result)
    #         print type(result_sum)
            target_sum = torch.sum(target)
            union = union + organWeight*(result_sum + target_sum) + (two*eps)
    
            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0
            
        IoU = intersect / union
#             out = torch.add(out, IoU.data*2)
        out =  one - two*IoU

    #     print type(out)
        return out   

'''
    This is dice loss for more than one organs, which means you can compute dice loss for more than one organ at a time, 
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
output:
    loss: Variable scalar
    
    succeed for multiple class segmentation problem
'''
def myDiceLoss4Organs(inputs, targets):
    
    eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
    one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
    two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))
                      
    inputSZ = inputs.size() #it should be sth like NxCxHxW
    
    inputs = F.softmax(inputs)
    _, results_ = inputs.max(1)
    results = torch.squeeze(results_) #NxHxW

    
    numOfCategories = inputSZ[1]
    
    ####### Convert categorical to one-hot format
    targetSZ = results.size() #NxHxW
    ## We consider NxHxW 3D tensor
#     result1 = torch.unsqueeze(results, 1) #Nx1xHxW
#     results_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_()) #NxCxHxW
#     results_one_hot.scatter_(1,result1,1) #scatter along the 'numOfDims' dimension
    results_one_hot = inputs
    
    
    target1 = Variable(torch.unsqueeze(targets.data,1)) #Nx1xHxW
    targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_()) #NxCxHxW
#     targets_one_hot = targets_one_hot.permute(0,2,3,1) #NxHxWxC
    targets_one_hot.scatter_(1, target1, 1) #scatter along the 'numOfDims' dimension
#     print 'line 367: one_hot size: ',targets_one_hot.size()
    
    
    ###### Now the prediction and target has become one-hot format
    ###### Compute the dice for each organ
#     intersects = Variable(torch.FloatTensor(numOfCategories).zero_())
#     unions = Variable(torch.FloatTensor(numOfCategories).zero_())
    out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad = True)
#     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
#     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)

    for organID in range(0, numOfCategories):
        target = targets_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
        result = results_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
#             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
        
#         intersect = torch.dot(result, target)
        intersect_vec = result * target
        intersect = torch.sum(intersect_vec)
#         print type(intersect)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
#         print type(result_sum)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (two*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
#             out = torch.add(out, IoU.data*2)
        out = out + one - two*IoU
#         intersects[organID], unions[organID] = intersect, union
#         print('organID: {} union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
# organID, union.data[0], intersect.data[0], target_sum.data[0], result_sum.data[0], IoU.data[0])

        
    denominator = Variable(torch.cuda.FloatTensor(1).fill_(numOfCategories))
    
    out = out / denominator
#     print type(out)
    return out


'''
    Function to calculate the Generalised Dice Loss defined in Sudre, C. et. al.
     (2017) Generalised Dice overlap as a deep learning loss function for highly
      unbalanced segmentations. DLMIA 2017
      The only issue I have is that it may only suits to two types
Input:
    prediction: the logits (before softmax)
    ground_truth: the segmentation ground truth
    weight_map:
    type_weight: type of weighting allowed between labels (choice
    between Square (square of inverse of volume), Simple (inverse of volume)
    and Uniform (no weighting))
Output:
    the loss
'''
def generalised_dice_loss(prediction, ground_truth, weight_map=None, type_weight='Square'):

    ground_truth = tf.to_int64(ground_truth)
    n_voxels = ground_truth.get_shape()[0].value
    n_classes = prediction.get_shape()[1].value
    prediction = tf.nn.softmax(prediction)
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels], dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])

    if weight_map is not None:
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\"" \
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = \
        tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    return 1 - generalised_dice_score



'''
    This is dice loss for one organ, which means you can compute dice for more than one organs at a time, 
    but it adapts to 0/1/2/3... (foreground,organ1,organ2,... problem)
input:
    inputs: predicted segmentation map, tensor type,  Variable (NxCxHxW)
    targets: real segmentation map, tensor type,  Variable (NxHxW)
output:
    loss: Variable scalar
    
    succeed for multiple organs
'''
   
class myWeightedDiceLoss4Organs(nn.Module):
    def __init__(self, organIDs = [1], organWeights=[1]):
        super(myWeightedDiceLoss4Organs, self).__init__()
        self.organIDs = organIDs
        self.organWeights = organWeights
#         pass

    def forward(self, inputs, targets, save=True):
        
        eps = Variable(torch.cuda.FloatTensor(1).fill_(0.000001))
        one = Variable(torch.cuda.FloatTensor(1).fill_(1.0))
        two = Variable(torch.cuda.FloatTensor(1).fill_(2.0))
                          
        inputSZ = inputs.size() #it should be sth like NxCxHxW
        
        inputs = F.softmax(inputs, dim=1)
#         _, results_ = inputs.max(1)
#         results = torch.squeeze(results_) #NxHxW
    
        numOfCategories = inputSZ[1]
        assert numOfCategories==len(self.organWeights), 'organ weights is not matched with organs (bg should be included)'
        ####### Convert categorical to one-hot format

        results_one_hot = inputs
        
        
        target1 = Variable(torch.unsqueeze(targets.data,1)) #Nx1xHxW
        targets_one_hot = Variable(torch.cuda.FloatTensor(inputSZ).zero_()) #NxCxHxW
        targets_one_hot.scatter_(1, target1, 1) #scatter along the 'numOfDims' dimension
 
        ###### Now the prediction and target has become one-hot format
        ###### Compute the dice for each organ
        out = Variable(torch.cuda.FloatTensor(1).zero_(), requires_grad = True)
    #     intersect = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
    #     union = Variable(torch.cuda.FloatTensor([1]).zero_(), requires_grad = True)
    
        for organID in range(0, numOfCategories):
            target = targets_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
            result = results_one_hot[:,organID,:,:].contiguous().view(-1,1).squeeze(1)
    #             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
            
    #         intersect = torch.dot(result, target)
            intersect_vec = result * target
            intersect = torch.sum(intersect_vec)
    #         print type(intersect)
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result)
    #         print type(result_sum)
            target_sum = torch.sum(target)
            union = result_sum + target_sum + (two*eps)
    
            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0
            IoU = intersect / union
    #             out = torch.add(out, IoU.data*2)
            out = out + self.organWeights[organID] * (one - two*IoU)
    #         print('organID: {} union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
    # organID, union.data[0], intersect.data[0], target_sum.data[0], result_sum.data[0], IoU.data[0])
    
        denominator = Variable(torch.cuda.FloatTensor(1).fill_(sum(self.organWeights)))    
#         denominator = Variable(torch.cuda.FloatTensor(1).fill_(numOfCategories))
        
        out = out / denominator
    #     print type(out)
        return out        


'''
    This is generalized dice loss for organs, which means you can compute dice for more than one organ at a time
    Warning: The backward cannot done automatically, and this kind of dice expansion to multi-class is not correct
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
    
output:
    loss: scalar
    
    so far, not succeed, but will be succeeded
'''
   
class GenDiceLoss4Organs(nn.Module):
    def __init__(self,  organIDs = [1], organWeights=[1], size_average=True):   
        super(DiceLoss4Organs,self).__init__()
        self.organIDs = organIDs
        self.organWeights = organWeights
        self.size_average = size_average
        
    def forward(self, inputs, targets):
#         loss = 0.0
#         loss = torch.autograd.Variable(torch.Tensor(1), requires_grad=True)
  
        eps = 0.000001
        totalWeight = np.sum(self.organWeights)
        
        _,results = inputs.max(1) #maximize the channels
        results = torch.squeeze(results)


        sz =targets.size()
        singleSZ = targets[0,...].size()
#         print type(sz),singleSZ
        diceLoss = 0.0
        loss = np.zeros(sz[0])
#         print '..... target size: ',targets.size()
#         print '..... results size: ',results.size()
#         print type(inputs)
        for k in range(0,sz[0]):
            result_ = torch.FloatTensor(singleSZ)
            target_ = torch.FloatTensor(singleSZ)
            result_.copy_(results.data[k])
            target_.copy_(targets.data[k])
            for ind in range(0,len(self.organIDs)):
                id = self.organIDs[ind]
                #get the single organ representation
    #             indI = np.where(inputs==id)
    #             indI = inputs:eq(id)
                result = torch.zeros(result_.size())
                result[result_==id] = 1
                
    #             indT = np.where(targets==id)
    #             indT = targets:eq(id)
                target = torch.zeros(target_.size())
                target[target_==id] = 1
                
    #             iflat = input.view(-1)
    #             tflat = target.view(-1)
                intersect = torch.dot(result,target) #no need to flatern input or target
                intersect = np.max([eps, intersect]) #scalar value
    #             print '......intersect is: ', intersect
                input_sum = torch.sum(result)
                target_sum = torch.sum(target)
                union = input_sum + target_sum + 2*eps #not the real union
    #             print '.......union is: ', union
    #             print '..........dice is: ', 1.0*self.organWeights[ind]/totalWeight * (2.0 * intersect / union)
                loss[k] += 1.0 * self.organWeights[ind]/totalWeight*(1.0 - (2.0 * intersect / union))
            
#         loss = loss/len(self.organIDs)
#         print type(loss)
#         print 'loss shape: ', loss.shape,' loss is ',loss
        if self.size_average:
            diceLoss = loss.mean()
        else:
            diceLoss = loss.sum()

        diceLoss = torch.cuda.FloatTensor(1).fill_(1*diceLoss)
        diceLoss = Variable(diceLoss,requires_grad=True)
#         print 'haha2',type(loss)
        
        return diceLoss

    


'''
    This is dice loss for one organ, which means you can only compute dice for one organ at a time, 
    but it only adapts to 0/1 (foreground,background problem)
    Note: we need to implement backward manually
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
   
output:
    loss: scalar
    
    so far, not succeed
'''
   
class WeightedDiceLoss4Organs(Function):
    def __init__(self, *args, **kwargs):
        self.numOfCategories = 2
#         pass

    def forward(self, inputs, targets, save=True):
        if save:
            self.save_for_backward(inputs, targets)
        eps = 0.000001
        
        inputSZ = inputs.size() #it should be sth like NxCxHxW
        
#         print 'line 336: inputs size: ',inputSZ
#         print 'line 337: input: ', inputs[1,1,1,1]
        
        _, results_ = inputs.max(1)
#         print 'line 339: results_ size: ',results_.size()
        results_ = torch.squeeze(results_) #NxHxW
#         print 'line 338: unique(results): ',unique(results_)
        self.inputs = torch.cuda.FloatTensor(inputs.size())
        self.inputs.copy_(inputs)
        
#         if inputs.is_cuda:
#             results = torch.cuda.FloatTensor(results_.size())
#             self.targets_ = torch.cuda.FloatTensor(targets.size())
#         else:
#             results = torch.FloatTensor(results_.size())
#             self.targets_ = torch.FloatTensor(targets.size())
            
        results = torch.cuda.LongTensor(results_.size())
        self.targets_ = torch.cuda.LongTensor(targets.size())
        results.copy_(results_) #NxHxW
        self.targets_.copy_(targets) 
        targets = self.targets_ #NxHxW
        
        self.numOfCategories = inputSZ[1]
        
        ####### Convert categorical to one-hot format
        targetSZ = results.size() #NxHxW
        numOfDims = len(targetSZ)
        ## We consider NxHxW 3D tensor
        result1 = torch.unsqueeze(results, numOfDims) #NxHxWx1
        self.results_one_hot = torch.cuda.FloatTensor(inputSZ).zero_() #NxCxHxW
        self.results_one_hot = self.results_one_hot.permute(0,2,3,1) #NxHxWxC
        self.results_one_hot.scatter_(numOfDims,result1,1) #scatter along the 'numOfDims' dimension
#         print 'line 361: one_hot size: ',self.results_one_hot.size()
        
        target1 = torch.unsqueeze(targets,numOfDims) #NxHxWx1
        self.targets_one_hot = torch.cuda.FloatTensor(inputSZ).zero_() #NxCxHxW
        self.targets_one_hot = self.targets_one_hot.permute(0,2,3,1) #NxHxWxC
        self.targets_one_hot.scatter_(numOfDims,target1,1) #scatter along the 'numOfDims' dimension
#         print 'line 367: one_hot size: ',self.targets_one_hot.size()
        
        
        ###### Now the prediction and target has become one-hot format
        ###### Compute the dice for each organ
#         out = torch.FloatTensor(self.numOfCategories).zero_()
        sumOut = 0.0
        self.intersect = torch.FloatTensor(self.numOfCategories).zero_()
        self.union = torch.FloatTensor(self.numOfCategories).zero_()
        for organID in range(0, self.numOfCategories):
            target = self.targets_one_hot[:,:,:,organID].contiguous().view(-1,1).squeeze(1)
            result = self.results_one_hot[:,:,:,organID].contiguous().view(-1,1).squeeze(1)
#             print 'unique(target): ',unique(target),' unique(result): ',unique(result)
            
            intersect = torch.dot(result, target)
            # binary values so sum the same as sum of squares
            result_sum = torch.sum(result)
            target_sum = torch.sum(target)
            union = result_sum + target_sum + (2*eps)

            # the target volume can be empty - so we still want to
            # end up with a score of 1 if the result is 0/0
            IoU = intersect / union
#             print('organID: {} union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#                 organID, union, intersect, target_sum, result_sum, 2*IoU))
#             out = torch.FloatTensor(1).fill_(2*IoU)
#             out[organID].fill_(2*IoU)
            sumOut = sumOut + IoU * 2
            self.intersect[organID], self.union[organID] = intersect, union
        out = torch.cuda.FloatTensor(1).fill_(sumOut)
        return out

    def backward(self, grad_output):
        inputs, _ = self.saved_tensors # we need probabilities for input
 
        targets = self.targets_one_hot # we need binary for targets
        intersects, unions = self.intersect, self.union
#         print 'targets size: ',targets.size(),'unions size: ',unions.size(),'intersects size: ',intersects.size()
        for i in range(0,self.numOfCategories):
            input = inputs[:,i,...]
            target = targets[...,i]
            union = unions[i]
            intersect = intersects[i]
            gt = torch.div(target, union)
            IoU2 = intersect/(union*union)
            print 'line 419: IoU2: ',IoU2

#             pred = torch.mul(input[:, 1], IoU2) #input[:,1] is equal to input[:,1,...]
            pred = torch.mul(input, IoU2) #input[:,1] is equal to input[:,1,...]
            print 'line 423: input: ',input.cpu()[1,1,1,...]
            print 'line 423: pred: ',pred.cpu()[1,1,1,...]
#             print 'gt size: ',gt.size(),' pred size: ',pred.size()
            print 'line 423: gt: ', gt.cpu()[1,1,1]
            dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
            if i==0:
                prev = torch.mul(dDice, grad_output[0])
            else:
                curr = torch.mul(dDice, -grad_output[0])
                grad_input = torch.cat((prev,curr), 0)
                prev = curr
        print 'line 429: grad_output: ',grad_output.cpu()        
        print 'line 430: grad_input: ', grad_input.cpu()[1,1,1,...]
        return grad_input , None

# def dice_loss(input, target):
#     return DiceLoss()(input, target)
# 
# def dice_error(input, target):
#     eps = 0.000001
#     _, result_ = input.max(1)
#     result_ = torch.squeeze(result_)
#     if input.is_cuda:
#         result = torch.cuda.FloatTensor(result_.size())
#         target_ = torch.cuda.FloatTensor(target.size())
#     else:
#         result = torch.FloatTensor(result_.size())
#         target_ = torch.FloatTensor(target.size())
#     result.copy_(result_.data)
#     target_.copy_(target.data)
#     target = target_
#     intersect = torch.dot(result, target)
# 
#     result_sum = torch.sum(result)
#     target_sum = torch.sum(target)
#     union = result_sum + target_sum + 2*eps
#     intersect = np.max([eps, intersect])
#     # the target volume can be empty - so we still want to
#     # end up with a score of 1 if the result is 0/0
#     IoU = intersect / union
# #    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
# #        union, intersect, target_sum, result_sum, 2*IoU))
#     return 2*IoU

def unique(tensor1d):
    t, idx = np.unique(tensor1d.cpu().numpy(), return_inverse=True)
    return t


'''
    This is dice loss for organs, which means you can compute dice for more than one organ at a time
    Warning: The backward cannot done automatically, and this kind of dice expansion to multi-class is not correct
    Thus, I rewrite it.
input:
    inputs: predicted segmentation map, tensor type, even Variable
    targets: real segmentation map, tensor type, even Variable
    
output:
    loss: scalar
    
    so far, not succeed
'''
   
class DiceLoss4Organs(Function):
    def __init__(self,  organIDs = [1], organWeights=[1], size_average=True):   
        super(DiceLoss4Organs,self).__init__()
        self.organIDs = organIDs
        self.organWeights = organWeights
        self.size_average = size_average
        
    def forward(self, inputs, targets):
#         loss = 0.0
#         loss = torch.autograd.Variable(torch.Tensor(1), requires_grad=True)
  
        eps = 0.000001
        totalWeight = np.sum(self.organWeights)
        
        _,results = inputs.max(1) #maximize the channels
        results = torch.squeeze(results)
#         if inputs.is_cuda:
#             print 'joking'
#             result_ = torch.cuda.FloatTensor(results.size())
#             target_ = torch.cuda.FloatTensor(targets.size())
#         else:
#             result_ = torch.FloatTensor(results.size())
#             target_ = torch.FloatTensor(targets.size())
#         print 'haha'

        sz =targets.size()
        singleSZ = targets[0,...].size()
#         print type(sz),singleSZ
        diceLoss = 0.0
        loss = np.zeros(sz[0])
#         print '..... target size: ',targets.size()
#         print '..... results size: ',results.size()
#         print type(inputs)
        for k in range(0,sz[0]):
            result_ = torch.FloatTensor(singleSZ)
            target_ = torch.FloatTensor(singleSZ)
            result_.copy_(results.data[k])
            target_.copy_(targets.data[k])
            for ind in range(0,len(self.organIDs)):
                id = self.organIDs[ind]
                #get the single organ representation
    #             indI = np.where(inputs==id)
    #             indI = inputs:eq(id)
                result = torch.zeros(result_.size())
                result[result_==id] = 1
                
    #             indT = np.where(targets==id)
    #             indT = targets:eq(id)
                target = torch.zeros(target_.size())
                target[target_==id] = 1
                
    #             iflat = input.view(-1)
    #             tflat = target.view(-1)
                intersect = torch.dot(result,target) #no need to flatern input or target
                intersect = np.max([eps, intersect]) #scalar value
    #             print '......intersect is: ', intersect
                input_sum = torch.sum(result)
                target_sum = torch.sum(target)
                union = input_sum + target_sum + 2*eps #not the real union
    #             print '.......union is: ', union
    #             print '..........dice is: ', 1.0*self.organWeights[ind]/totalWeight * (2.0 * intersect / union)
                loss[k] += 1.0 * self.organWeights[ind]/totalWeight*(1.0 - (2.0 * intersect / union))
            
#         loss = loss/len(self.organIDs)
#         print type(loss)
#         print 'loss shape: ', loss.shape,' loss is ',loss
        if self.size_average:
            diceLoss = loss.mean()
        else:
            diceLoss = loss.sum()

        diceLoss = torch.cuda.FloatTensor(1).fill_(1*diceLoss)
        diceLoss = Variable(diceLoss,requires_grad=True)
#         print 'haha2',type(loss)
        
        return diceLoss
    
    def backward(self, grad_ouput):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input , None
    

'''
compute the attention weights: 
    for prob belongs to [0.3,0.7] we use positive weights; 
    and we use negative weights belonging to [0,0.3) and (0.7,1]
Inputs:
    prob: probability
Outputs:
    res: attention we compute
'''
def computeAttentionWeight(prob):
    if prob>0.5:
        prob = 1 - prob
    res = 10.0/3*prob*prob + 7.0/3*prob -1
    if prob<0.3:
        res = 0
    return res


'''
compute the attention weights: 
    for prob belongs to [0.3,0.7] we use negative weights (easy samples for the segmenter, hard samples for the discriminator); 
    and we use positive weights belonging to [0,0.3) and (0.7,1] (hard samples for the segmenter, easy samples for the discriminator).
Inputs:
    prob: probability
Outputs:
    res: attention we compute
'''
def computeSampleAttentionWeight(prob):
    res = 12.5*(prob-0.5)*(prob-0.5)-0.5
    return res


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=8):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)   
    
    
'''
    Sets the learning rate to the initial LR decayed by 10 every 10 epochs
'''
def adjust_learning_rate(optimizer, lr):
#     lr = opt.lr * (0.1 ** (epoch // opt.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  
    return lr
