
import torch
from torch import nn
from torch.nn import init

import pretrainedmodels as pm
import torchvision.models as torchvis_models
import sys
import numpy as np
# Pseudocode, omitting the specific implementation of (interpolate, extract_patches, tup2list, hw_flatten)
from utils import (interpolate, extract_patches, tup2list, hw_flatten)
FLAGS = None # Pseudocode, omitting the specific implementation of FLAGS

PROJECT_ROOT = 'path/project/root'

class ResChanAttnMultiScaleAttn(nn.Module):
    def __init__(self, in_dim, out_img_res, start_CHW=(64, 14, 14), out_chan_ls=[64, 64, 64], n_chan_output=3, G=2, branch_type = 'assp_branch'):
        super(ResChanAttnMultiScaleAttn, self).__init__()
        self.start_CHW = start_CHW
        self.upsample_scale_factor = int((out_img_res / start_CHW[-1]) ** (1/len(out_chan_ls)))
        self.input_fc = nn.Linear(in_dim, np.prod(self.start_CHW))
        self.G = G
        gn_chan = int(start_CHW[0] // self.G)
        scales = 4
        n_chan = start_CHW[0]

        kernel_size = 5
        pad_size = int(kernel_size // 2)
        assert all(x == out_chan_ls[0] for x in out_chan_ls)
        self.blocks = []
        self.blocks.append(ResUnchBlock(start_CHW[0], start_CHW[0], kernel_size=5))
        self.blocks.append(ResUnchBlock(start_CHW[0], start_CHW[0], kernel_size=5))
        
        self.blocks.append(ASSPMulitScale(in_channels = n_chan, output_stride = scales, out_channels = n_chan, GN_chan = gn_chan, branch_type = branch_type))
        self.blocks.append(MultiScaleAttentionSK(channel = n_chan, scales = scales + 1))
        self.blocks.append(ChannelAttentionECA(kernel_size = 3, G=self.G))
        
        for ind, _ in enumerate(out_chan_ls):
            if ind == 0:
                in_chan = start_CHW[0]
                out_chan = out_chan_ls[0]
            else:
                in_chan = out_chan_ls[ind-1]
                out_chan = out_chan_ls[ind]
            self.blocks.append(ResUpBlock(in_chan, out_chan, kernel_size=5, upsample_scale_factor = self.upsample_scale_factor))
            self.blocks.append(ResUnchBlock(out_chan, out_chan, kernel_size=5))

            self.blocks.append(ASSPMulitScale(in_channels = n_chan, output_stride = scales, out_channels = n_chan, GN_chan = gn_chan, branch_type = branch_type))
            self.blocks.append(MultiScaleAttentionSK(channel = n_chan, scales = scales + 1))
            self.blocks.append(ChannelAttentionECA(kernel_size = 3, G=self.G))

        self.blocks = nn.Sequential(*self.blocks)

        pad_size = int(kernel_size // 2)
        self.top = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Conv2d(out_chan_ls[-1], n_chan_output, kernel_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.input_fc(x)
        x = x.view(-1, *self.start_CHW)

        for ind, block in enumerate(self.blocks):
            x = block(x)
        x = self.top(x)
        return x
def assp_branch(in_channels, out_channels, kernel_size, dilation, GN_chan = 16):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(GN_chan, out_channels)
            # nn.ReLU(inplace=True)
            )

def ks_branch(in_channels, out_channels, kernel_size, GN_chan = 16):
    # padding = 0 if kernel_size == 1 else dilation
    padding = int(kernel_size // 2)
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(GN_chan, out_channels)
            # nn.ReLU(inplace=True)
            )
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
class ASSPMulitScale(nn.Module):
    def __init__(self, in_channels, output_stride, assp_channels=4, out_channels = 256, GN_chan = 16, branch_type = 'assp_branch'):
        super(ASSPMulitScale, self).__init__()

        assert output_stride in [4, 8], 'Only output strides of 8 or 16 are suported'
        assert assp_channels in [4, 6], 'Number of suported ASSP branches are 4 or 6'
        dilations = [1, 2, 5, 7, 9, 11]
        kss = [1, 3, 5, 7, 9, 11]
        dilations = dilations[:assp_channels]
        self.assp_channels = assp_channels
        if branch_type == 'assp_branch':
            self.aspp1 = assp_branch(in_channels, out_channels, 1, dilation=dilations[0], GN_chan = GN_chan)
            self.aspp2 = assp_branch(in_channels, out_channels, 3, dilation=dilations[1], GN_chan = GN_chan)
            self.aspp3 = assp_branch(in_channels, out_channels, 3, dilation=dilations[2], GN_chan = GN_chan)
            self.aspp4 = assp_branch(in_channels, out_channels, 3, dilation=dilations[3], GN_chan = GN_chan)
            if self.assp_channels == 6:
                self.aspp5 = assp_branch(in_channels, out_channels, 3, dilation=dilations[4], GN_chan = GN_chan)
                self.aspp6 = assp_branch(in_channels, out_channels, 3, dilation=dilations[5], GN_chan = GN_chan)
        elif branch_type == 'ks_branch':
            self.aspp1 = ks_branch(in_channels, out_channels, kernel_size=kss[0], GN_chan = GN_chan)
            self.aspp2 = ks_branch(in_channels, out_channels, kernel_size=kss[1], GN_chan = GN_chan)
            self.aspp3 = ks_branch(in_channels, out_channels, kernel_size=kss[2], GN_chan = GN_chan)
            self.aspp4 = ks_branch(in_channels, out_channels, kernel_size=kss[3], GN_chan = GN_chan)
            if self.assp_channels == 6:
                self.aspp5 = ks_branch(in_channels, out_channels, kernel_size=kss[4], GN_chan = GN_chan)
                self.aspp6 = ks_branch(in_channels, out_channels, kernel_size=kss[5], GN_chan = GN_chan)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.GroupNorm(GN_chan, out_channels)
            # nn.ReLU(inplace=True)
            )
        # self.conv1 = nn.Conv2d(out_channels*(self.assp_channels + 1), out_channels, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        if self.assp_channels == 6:
            x5 = self.aspp5(x)
            x6 = self.aspp6(x)
        x_avg_pool = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        # x_avg_pool = self.conv_temp(x)
        if self.assp_channels == 6:
            # x = self.conv1(torch.cat((x1, x2, x3, x4, x5, x6, x_avg_pool), dim=1))
            x = [x1, x2, x3, x4, x5, x6, x_avg_pool]
        else:
            # x = self.conv1(torch.cat((x1, x2, x3, x4, x_avg_pool), dim=1))
            x = [x1, x2, x3, x4, x_avg_pool]
        # x = self.bn1(x)
        # x = self.dropout(self.relu(x))

        return x
class MultiScaleAttentionSK(nn.Module):
    def __init__(self, channel=64,scales=4, kernel_size=3, reduction=4,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.attention_weights = None
        # self.convs=nn.ModuleList([])
        # for k in scales:
        #     self.convs.append(
        #         nn.Sequential(OrderedDict([
        #             ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
        #             ('bn',nn.BatchNorm2d(channel)),
        #             ('relu',nn.ReLU())
        #         ]))
        #     )
        # self.fc=nn.Linear(channel, self.d)
        self.conv1ds=nn.ModuleList([])
        for i in range(scales):
            # self.fcs.append(nn.Linear(self.d,channel))
            self.conv1ds.append(nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2))
        self.softmax=nn.Softmax(dim=0)

        ''' refer ASSP '''
        # self.conv1 = nn.Conv2d(channel*(scales + 1), channel, 1, bias=False)
        self.conv1 = nn.Conv2d(channel*(scales), channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
    def forward(self, conv_outs):
        # bs, c, _, _ = x.size() # bs,channel,h,w
        # conv_outs=[]
        ### split
        # for conv in self.convs: # k
        #     conv_outs.append(conv(x))
        feats=torch.stack(conv_outs, 0) # k,bs,channel,h,w
        k, bs, c, w, h = feats.size() # bs,channel,h,w
        ### fuse
        U=sum(conv_outs) # bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1).unsqueeze(1) # bs,c,h,w -> bs,c -> bs,1,c
        #bs,1,c
        # Z=self.fc(S) # bs,d

        ### calculate attention weight
        weights=[]
        for conv1d in self.conv1ds:
            weight=conv1d(S)
            weights.append(weight.view(bs,c,1,1)) # bs,channel
        attention_weights=torch.stack(weights,0) # k,bs,channel,1,1
        attention_weights=self.softmax(attention_weights) # k,bs,channel,1,1
        self.attention_weights = attention_weights
        ### fuse
        # v = (attention_weughts * feats).view(-1, c, w, h)
        v = (attention_weights * feats)
        v_ls = torch.cat([v[i] for i in range(v.size(0))], dim=1)

        V = self.bn1(self.conv1(v_ls))
        # V=(attention_weughts*feats).sum(0)
        return V      
class ChannelAttentionECA(nn.Module):

    def __init__(self, kernel_size=3, G = 2):
        super().__init__()
        self.G = G
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        # x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w
        # x_0 = x

        #channel attention
        x_channel=self.avg_pool(x) #bs*G,c,1,1
        x_channel=x_channel.squeeze(-1).permute(0,2,1) #bs*G,1,c
        x_channel=self.conv(x_channel) #bs*G,1,c
        x_channel=self.sigmoid(x_channel) #bs*G,1,c
        x_channel=x_channel.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        out = x*x_channel

        out = out.contiguous().view(b,-1,h,w)
        return out
  
class ResUnchBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=5):
        super(ResUnchBlock, self).__init__()
        pad_size = int(kernel_size // 2) # do not change H and W
        self.block = nn.Sequential(
            # nn.Upsample(scale_factor=upsample_scale_factor, mode='bicubic'),
            nn.ReflectionPad2d(pad_size),
            # nn.Conv2d(in_channels, out_channels, kernel_size)
            nn.Conv2d(in_chan, out_chan, kernel_size),
            nn.GroupNorm(32, out_chan),
            MemoryEfficientSwish(),
        )
    def forward(self, x):
        out = self.block(x)
        out += x
        return out

class ResUpBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=5, upsample_scale_factor=2, GN_chan = 32):
        super(ResUpBlock, self).__init__()
        pad_size = int(kernel_size // 2) # do not change H and W
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=upsample_scale_factor, mode='bicubic'),
            nn.ReflectionPad2d(pad_size),
            # nn.Conv2d(in_channels, out_channels, kernel_size)
            nn.Conv2d(in_chan, out_chan, kernel_size),
            nn.GroupNorm(GN_chan, out_chan),
            MemoryEfficientSwish(),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=upsample_scale_factor, mode='bicubic'), # change H and W
            # nn.Conv2d(in_chan, out_chan, stride=1, kernel_size=1), # change channel
            nn.GroupNorm(GN_chan, out_chan),
        )
    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        return out

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i))) 
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)   
     
class SeparableEncoderVGG19ml(nn.Module):
    def __init__(self, out_dim, random_crop_pad_percent, spatial_out_dim=None, drop_rate=0.25):
        super(SeparableEncoderVGG19ml, self).__init__()
        # out_dim: 4643
        self.drop_rate = drop_rate
        # for t in  torchvis_models.__dict__:
        #     print(t)
        
        if FLAGS.is_rgbd:
            bbn = torchvis_models.__dict__['vgg19'](pretrained=True)
        else:
            bbn = pm.__dict__['vgg19'](num_classes=1000, pretrained='imagenet')

        if FLAGS.is_rgbd:
            if FLAGS.is_rgbd == 1:  # RGBD
                bbn.features = nn.Sequential(
                    nn.Conv2d(4, 64, 3, padding=1),
                    *bbn.features[1:])
                ckpt_name = 'vgg19_rgbd_large_norm_within_img'
            else:  # Depth only
                bbn.features = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1),
                    *bbn.features[1:])
                ckpt_name = 'vgg19_depth_only_large_norm_within_img'

            state_dict_loaded = torch.load(f'{PROJECT_ROOT}/data/imagenet_rgbd/{ckpt_name}_best.pth.tar')['state_dict']
            state_dict_loaded = { k.replace('module.', ''): v for k, v in state_dict_loaded.items() }
            bbn.load_state_dict(state_dict_loaded)

            branch_dict = {  # VGG19 Blocks  # selectedLayers = [3, 6, 10, 14, 18], before maxpool
                'conv1': ['features.{}'.format(i) for i in range(4)],
                'conv2': ['features.{}'.format(i) for i in range(9)],
                'conv3': ['features.{}'.format(i) for i in range(18)],
                'conv4': ['features.{}'.format(i) for i in range(27)],
            }
        else:
            branch_dict = {  # VGG19 Blocks  # selectedLayers = [3, 6, 10, 14, 18], before maxpool
                'conv1': ['_features.{}'.format(i) for i in range(4)],
                'conv2': ['_features.{}'.format(i) for i in range(9)],
                'conv3': ['_features.{}'.format(i) for i in range(18)],
                'conv4': ['_features.{}'.format(i) for i in range(27)],
            }
        # e.g. 'conv1': ['_features.0', '_features.1', '_features.2', '_features.3']
        
        spatial_out_dims = None
        main_branch = list(branch_dict.values())[-1] 
        # main_branch: 27 features
        
        
        branch_dict = {layer: branch_module_list[-1] for layer, branch_module_list in branch_dict.items()}
        # branch_dict: {'conv1': '_features.3', 'conv2': '_features.8', 'conv3': '_features.17', 'conv4': '_features.26'}
        # sys.exit()
        
        self.multi_branch_bbn = MultiBranch(bbn, branch_dict, main_branch, spatial_out_dims=spatial_out_dims)

        
        self.bbn_n_out_planes = self.multi_branch_bbn.num_output_planes()
        self.patch_size = 3
        self.out_shapes = [(32, 28 - self.patch_size + 1, 28 - self.patch_size + 1)] * 3 + [(32, 14 - self.patch_size + 1, 14 - self.patch_size + 1)]
        self.n_out_planes = self.out_shapes[0][0]
        kernel_size = 3
        pad_size = int(kernel_size // 2)

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[0]),

            nn.MaxPool2d(2),

            nn.Conv2d(self.bbn_n_out_planes[0], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[1]),

            nn.Conv2d(self.bbn_n_out_planes[1], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[2]),

            nn.Conv2d(self.bbn_n_out_planes[2], self.n_out_planes, kernel_size, stride=2, padding=pad_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )

        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(self.bbn_n_out_planes[3]),

            nn.Conv2d(self.bbn_n_out_planes[3], self.n_out_planes, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.n_out_planes),
        )
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]

        
        # Separable part
        self.space_maps = nn.ModuleDict({
            str(in_space_dim): nn.Linear(in_space_dim**2, out_dim, bias=False) for in_space_dim in np.unique(tup2list(self.out_shapes, 1))
        })

        self.chan_mixes = nn.ModuleList([ChannelMix(self.n_out_planes*self.patch_size**2, out_dim) for _ in range(len(self.convs))])
        self.branch_mix = nn.Parameter(torch.Tensor(out_dim, len(self.chan_mixes)))
        self.branch_mix.data.fill_(1.)

        self.dropout = nn.Dropout(drop_rate)

        self.trainable = self.convs + list(self.space_maps.values()) + list(self.chan_mixes) + [self.branch_mix]

    def forward_bbn(self, x, detach_bbn=False):

        
        x = interpolate(x, size=int(FLAGS.im_res), mode=FLAGS.interp_mode)
        
        
        X = self.multi_branch_bbn(x)
        if detach_bbn:
            X = [xx.detach() for xx in X]
        feats_dict = dict(zip(self.multi_branch_bbn.branch_dict.keys(), X))
        
        return feats_dict

    def forward_convs(self, feats_dict):
        X = [conv(xx) for xx, conv in zip(feats_dict.values(), self.convs)]
        return X

    def forward(self, x, feats=False, detach_bbn=False):
        feats_dict = self.forward_bbn(x, detach_bbn=detach_bbn)
        X = self.forward_convs(feats_dict)
        X = [extract_patches(x, self.patch_size) for x in X]
        X = [self.space_maps[str(x.shape[-1])](hw_flatten(x)) for x in X]  # => BxCxV
        X = [self.dropout(x) for x in X]
        X = [f(x) for f, x in zip(self.chan_mixes, X)]

        x = torch.stack(X, dim=-1)
        x = (x * self.branch_mix.abs()).sum(-1)
        if feats:
            return x, feats_dict
        else:
            return x

class MultiBranch(nn.Module):
    def __init__(self, model, branch_dict, main_branch, spatial_out_dims=20, replace_maxpool=False):
        super(MultiBranch, self).__init__()
        name_to_module = dict(model.named_modules())
        self.branch_dict = branch_dict
        self.target_modules = list(branch_dict.values())
        self.main_branch = main_branch
        self.adapt_avg_pool_suffix = '_adapt_avg_pool'
        if spatial_out_dims is not None and isinstance(spatial_out_dims, int):
            spatial_out_dims = dict(zip(self.target_modules, [spatial_out_dims] * len(self.target_modules)))

        for module_name in main_branch:
            module = name_to_module[module_name]
            if replace_maxpool and isinstance(module, nn.MaxPool2d):
                module = nn.Upsample(scale_factor=.5)
            self.add_module(module_name.replace('.', '_'), module)
        for module_name in self.target_modules:
            if spatial_out_dims is not None:
                module = nn.AdaptiveAvgPool2d(spatial_out_dims[module_name])
                self.add_module(module_name + self.adapt_avg_pool_suffix, module)

    def __getitem__(self, module_name):
        return getattr(self, module_name.replace('.', '_'), None)

    def num_output_planes(self):
        n_planes = []
        for target_module in self.target_modules:
            for module_name in self.main_branch[:self.main_branch.index(target_module)+1][::-1]:
                try:
                    n_planes.append(list(self[module_name].parameters())[0].shape[0])
                    break
                except:
                    pass
        return n_planes

    def forward(self, x):
        X = {}
        for module_name in self.main_branch:
            if isinstance(self[module_name], nn.Linear) and x.ndim > 2:
                x = x.view(len(x), -1)
            x = self[module_name](x)
            if module_name in self.target_modules: # Collect
                X[module_name] = x.clone()
                avg_pool = self[module_name + self.adapt_avg_pool_suffix]
                if avg_pool:
                    X[module_name] = avg_pool(X[module_name])
        return list(X.values())

class ChannelMix(nn.Module):
    def __init__(self, n_chan, out_dim):
        super(ChannelMix, self).__init__()
        self.chan_mix = nn.Parameter(torch.Tensor(out_dim, n_chan))
        nn.init.xavier_normal(self.chan_mix)

        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.bias.data.fill_(0.01)

    def forward(self, x):
        # BxCxN
        x = (x * self.chan_mix.T).sum(-2)
        x += self.bias
        # BxN
        return x
    
def make_model(model_type, *args, **kwargs):
    return globals()[model_type](*args, **kwargs)
