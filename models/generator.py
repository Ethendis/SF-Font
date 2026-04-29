import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import scipy.io as io
import math
import numpy as np

try:
    from models.blocks import LinearBlock, Conv2dBlock, ResBlocks
except:
    from blocks import LinearBlock, Conv2dBlock, ResBlocks

import sys
sys.path.append('..')
from modules import modulated_deform_conv  

class Generator(nn.Module):   
    def __init__(self, img_size=80, sty_dim=64, n_res=2, use_sn=False):
        super(Generator, self).__init__()
        print("Init Generator")

        self.nf = 64 
        self.nf_mlp = 256

        self.decoder_norm = 'adain'

        self.adaptive_param_getter = get_num_adain_params
        self.adaptive_param_assign = assign_adain_params

        print("GENERATOR NF : ", self.nf)

        s0 = 16
        n_downs = 2
        nf_dec = 256

        self.cnt_encoder = ContentNet(self.nf, n_downs, n_res, 'in', 'relu', 'reflect')
        self.decoder = Mix(nf_dec, sty_dim, n_downs, n_res, self.decoder_norm, self.decoder_norm, 'relu', 'reflect', use_sn=use_sn)
        self.mlp = MLP(sty_dim, self.adaptive_param_getter(self.decoder), self.nf_mlp, 3, 'none', 'relu')
        self.skeleton = SkeletonNet()
        self.fusion = Multi_Head_Attention(num_channels = 256, n_heads=8, dropout=0.)

        self.apply(weights_init('kaiming'))

    def forward(self, x_src, s_ref):
        c_src, skip1, skip2 = self.cnt_encoder(x_src)

        x_out = self.decode(c_src, s_ref, skip1, skip2)
        return x_out

    def decode(self, cnt, sty, skip1, skip2):
        adapt_params = self.mlp(sty)
        self.adaptive_param_assign(adapt_params, self.decoder)
        out = self.decoder(cnt, skip1, skip2)
        return out
    def skeleton_feature(self,x):
        re = self.skeleton(x)
        return re
    def dusion(self,x, y):
        result = self.fusion(x,y)
        return result


    def _initialize_weights(self, mode='fan_in'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


class Mix(nn.Module):
    def __init__(self, nf_dec, sty_dim, n_downs, n_res, res_norm, dec_norm, act, pad, use_sn=False):
        super(Mix, self).__init__()
        print("Init Decoder")

        nf = nf_dec
        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, nf, res_norm, act, pad, use_sn=use_sn))

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(nf, nf//2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(2*nf, nf//2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(Conv2dBlock(2*nf, 3, 7, 1, 3, norm='none', act='tanh', pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        self.dcn = modulated_deform_conv.ModulatedDeformConvPack(64, 64, kernel_size=(3, 3), stride=1, padding=1, groups=1, deformable_groups=1, double=True).cuda()
        self.dcn_2 = modulated_deform_conv.ModulatedDeformConvPack(128, 128, kernel_size=(3, 3), stride=1, padding=1, groups=1, deformable_groups=1, double=True).cuda()

    def forward(self, x, skip1, skip2):
        output = x
        #print(x.shape)
        # for i in range(256):
        #     import matplotlib.pyplot as plt
        # # plt.figure()
        # # plt.savefig('./x[0][{}].cpu().numpy()'.format(i))
        #     plt.imshow(x[0][i].cpu().numpy())
        #     # plt.show()
        #     plt.savefig("./feature/fusion/fusion_{}.jpeg".format(i))
        #
        #     # plt.savefig('pic{}.jpg'.format(i), bbox_inches='tight')
        #     # plt.show()
        #     # print(x.shape)
        # exit()
        # print("model:",len(self.model))

        # print(output.shape)
        # for i in range(256):
        #     import matplotlib.pyplot as plt
        #     # plt.figure()
        #     # plt.savefig('./x[0][{}].cpu().numpy()'.format(i))
        #     plt.imshow(x[0][i].cpu().numpy())
        #     # plt.show()
        #     plt.savefig("./feature/res/res3_{}.jpeg".format(i))
        #
        #     # plt.savefig('pic{}.jpg'.format(i), bbox_inches='tight')
        #     # plt.show()
        # exit()
        for i in range(len(self.model)):
            output = self.model[i](output)
        #     print(output.shape)
        #     for i in range(733):
        #         import matplotlib.pyplot as plt
        #         # plt.figure()
        #         # plt.savefig('./x[0][{}].cpu().numpy()'.format(i))
        #         plt.imshow(x[0][i].cpu().numpy())
        #         # plt.show()
        #         plt.savefig("./feature/za/za_{}.jpeg".format(i))

                # plt.savefig('pic{}.jpg'.format(i), bbox_inches='tight')
                # plt.show()
                # print(x.shape)

            if i == 2:
                deformable_concat = torch.cat((output,skip2), dim=1)
                concat_pre, offset2 = self.dcn_2(deformable_concat, skip2)
                #print(concat_pre.shape)
                #print(x.shape)
                output = torch.cat((concat_pre,output), dim=1)

            if i == 4:
                deformable_concat = torch.cat((output,skip1), dim=1)
                concat_pre, offset1 = self.dcn(deformable_concat, skip1)
                #print(concat_pre.shape)

                output = torch.cat((concat_pre,output), dim=1)
                #print(output.shape)


        offset_sum1 = torch.mean(torch.abs(offset1))
        offset_sum2 = torch.mean(torch.abs(offset2))
        offset_sum = (offset_sum1+offset_sum2)/2
        return output, offset_sum


class ContentNet(nn.Module):
    def __init__(self, nf_cnt, n_downs, n_res, norm, act, pad, use_sn=False):
        super(ContentNet, self).__init__()
        print("Init ContentEncoder")

        nf = nf_cnt

        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, 256, norm=norm, act=act, pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        self.dcn1 = modulated_deform_conv.ModulatedDeformConvPack(3, 64, kernel_size=(7, 7), stride=1, padding=3, groups=1, deformable_groups=1).cuda()
        self.dcn2 = modulated_deform_conv.ModulatedDeformConvPack(64, 128, kernel_size=(4, 4), stride=2, padding=1, groups=1, deformable_groups=1).cuda()
        self.dcn3 = modulated_deform_conv.ModulatedDeformConvPack(128, 256, kernel_size=(4, 4), stride=2, padding=1, groups=1, deformable_groups=1).cuda()
        self.IN1 = nn.InstanceNorm2d(64)
        self.IN2 = nn.InstanceNorm2d(128)
        self.IN3 = nn.InstanceNorm2d(256)
        self.activation = nn.ReLU(inplace=True)


    def forward(self, x):
        x, _ = self.dcn1(x, x)
        x = self.IN1(x)
        x = self.activation(x)
        skip1 = x

        x, _ = self.dcn2(x, x)
        x = self.IN2(x)

        x = self.activation(x)
        skip2 = x

        x, _ = self.dcn3(x, x)
        x = self.IN3(x)
        x = self.activation(x)
        #x = self.att8(x)

        x = self.model(x)
        # for i in range(256):
        #     # plt.figure()
        #     # plt.savefig('./x[0][{}].cpu().numpy()'.format(i))
        #     plt.imshow(x[0][i].cpu().numpy())
        #     # plt.show()
        #     plt.savefig("./feature/model1/amodel_{}.jpeg".format(i))
        #
        #     # plt.savefig('pic{}.jpg'.format(i), bbox_inches='tight')
        #     # plt.show()
        #     # print(x.shape)

        return x, skip1, skip2

import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, nf_in, nf_out, nf_mlp, num_blocks, norm, act, use_sn=False):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp
        self.model.append(LinearBlock(nf_in, nf, norm=norm, act=act, use_sn=use_sn))
        for _ in range(num_blocks - 2):
            self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))
        self.model.append(LinearBlock(nf, nf_out, norm='none', act='none', use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class SkeletonNet(nn.Module):
    def __init__(self, norm = "in", act ="relu", pad = 'reflect', use_sn=False):
        super(SkeletonNet, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(ResBlocks(2, 256, norm=norm, act=act, pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        self.con1 = nn.Conv2d(3,64,7,1,3)
        self.con2 = nn.Conv2d(64,128,4,2,1)
        self.con3 = nn.Conv2d(128,256, 4, 2, 1)
        self.IN1 = nn.InstanceNorm2d(64)
        self.IN2 = nn.InstanceNorm2d(128)
        self.IN3 = nn.InstanceNorm2d(256)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.con1(x)
        x = self.IN1(x)
        x = self.activation(x)
        x = self.con2(x)
        x = self.IN2(x)
        x = self.activation(x)
        x = self.con3(x)
        x = self.IN3(x)
        x = self.activation(x)
        x = self.model(x)
        return x

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg19cut': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'N'],
}


class StyleNet(nn.Module):
    def __init__(self, img_size=64, output_k={'cont': 128, 'disc': 10}):
        super(StyleNet, self).__init__()
        # network layers setting
        self.features = make_layers(cfg['vgg11'], True)

        self.disc = nn.Linear(512, output_k['disc'])
        self.cont = nn.Linear(512, output_k['cont'])

        self._initialize_weights()

    def forward(self, x, sty=False):
        x = self.features(x)
        #print(x.shape)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        cont = self.cont(flat)
        if sty:
            return cont
        disc = self.disc(flat)
        return {'cont': cont, 'disc': disc}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def moco(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        cont = self.cont(flat)
        return cont

    def iic(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        flat = x.view(x.size(0), -1)
        disc = self.disc(flat)
        return disc

class Multi_Head_Attention(nn.Module):
    def __init__(self, num_channels = 256, n_heads=8, dropout=0.):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = n_heads
        self.num_channels = num_channels
        self.linears_key = nn.Linear(num_channels, num_channels)
        self.linears_value = nn.Linear(num_channels, num_channels)
        self.linears_query = nn.Linear(num_channels,num_channels)

        self.fc = nn.Linear(num_channels, num_channels)

        self.layer_norm = nn.LayerNorm(num_channels)
    def Reshape_Linear(self, num_channels = 256, n_heads=8, dropout=0.):
        self.linears_key = nn.Linear(num_channels, num_channels)
        self.linears_value = nn.Linear(num_channels, num_channels)
        self.linears_query = nn.Linear(num_channels, num_channels)
    def forward(self, x,y):
        content_permute = x.permute(0, 2, 3, 1)  # B,H,W,C
        batch, h, w, channel = content_permute.shape
        d_channel = int(channel / self.num_heads)
        content_feats_reshape = torch.reshape(content_permute, (batch, h * w, channel))  # B, HW, C
        # print(content_permute.shape,batch,h,w,channel,d_channel,content_feats_reshape.shape)

        query_matrix = self.linears_query(content_feats_reshape)

        # print(query_matrix.shape)

        residual = query_matrix
        query_matrix = query_matrix.view(batch, h * w, self.num_heads, d_channel)  # [B, HW, num_heads, C/num_heads]
        # print(query_matrix.shape)
        query_matrix = query_matrix.permute(0, 2, 3, 1)  # [B,num_heads,C/num_head,HW

        # 骨架k、v
        content_skeleton_permute = y.permute(0, 2, 3, 1)  # B,H,W,C
        batch, h, w, channel = content_skeleton_permute.shape

        content_skeleton_reshape = torch.reshape(content_skeleton_permute, (batch, h * w, channel))  # B, HW, C
        query_matrix_skeleton = self.linears_value(content_skeleton_reshape)
        query_matrix_skeleton = query_matrix_skeleton.view(batch, h * w, self.num_heads, d_channel)
        # print(query_matrix.shape)
        # exit()

        key_matrix = query_matrix_skeleton.permute(0, 2, 3, 1)  #
        v_matrix = query_matrix_skeleton.permute(0, 2, 1, 3)

        # print(key_matrix.shape)
        ######### attention ########
        # softmax & square root
        # [B, num_heads, HW, 3HW]
        attention_mask = torch.matmul(query_matrix.permute(0, 1, 3, 2), key_matrix)  # [B, num_heads, HW, HW]
        # print(key_matrix.shape, query_matrix.shape, attention_mask.shape)
        # exit()
        attention_mask = attention_mask.permute(0, 1, 3, 2) / math.sqrt(
            h * w)  # [B, num_heads, HW, 3HW]#除以系数
        # print(attention_mask.shape,111)
        # exit()
        attention_mask = F.softmax(attention_mask, dim=-1)
        # print(attention_mask.shape)
        # exit()
        # [B, num_heads, C/num_heads, HW]
        value_mask = torch.matmul(attention_mask, v_matrix)  # [B, num_heads, HW, C/num_heads]
        # print(value_mask.shape,122)
        # exit()
        value_mask = value_mask.permute(0, 1, 3, 2)  # [B, num_heads, C/num_heads, HW]
        value_mask = torch.reshape(value_mask, (batch, channel, -1))

        value_mask = value_mask.view(batch, h * w, self.num_channels)  # [B, HW, C]

        value_mask = self.fc(value_mask)
        # value_mask += residual  # [B, HW, C]

        value_mask = self.layer_norm(value_mask)
        value_mask = value_mask.view(batch, self.num_channels, h * w)
        # print(value_mask.shape)
        feat_scs = value_mask.view(batch, self.num_channels, h, w)
        # print(feat_scs.shape)
        return feat_scs


from torch import nn
import torch.nn.functional as F

try:
    from models.blocks import Conv2dBlock, FRN
except:
    from blocks import Conv2dBlock, FRN



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:
            #conv2d = Dynamic_conv2d(in_channels, v, kernel_size=3, padding=1)
            # from Multi_con import MultiScaleBlock
            # conv2d = MultiScaleBlock(in_channels,v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]

            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)




def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            num_adain_params += 2*m.num_features
    return num_adain_params






if __name__ =="__main__":
    pass
    # x_in = torch.randn(1, 256, 64, 64)
    # y_in = torch.randn(1, 256, 64, 64)
    # a = Attention(x_in,y_in)
    # print(a.shape)
    # import cv2
    # img_list = []
    # for i in range(1):
    #     img = cv2.imread("../data/font0/000{}.png".format(i))
    #
    #
    #     #print(img.shape)#256*256*3
    #     img = torch.tensor(img)
    #     img = img.permute(2, 0, 1)
    #     img_list.append(img)
    #     print(img.shape)
    #
    # img_tensor = torch.stack(img_list)
    # print(type(img_tensor))
    # print(img_tensor.shape)
    #
    #
    #
    #
    #
    # def imshow(img):
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #
    #     # img = img / 2 + 0.5  # 反标准化
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    #
    # import torchvision
    #
    # imshow(torchvision.utils.make_grid(img_tensor))
    #
    # S = Skeleton()
    #
    # plt.show()
    #
    # x_in =torch.randn(4,3,256,256)
    # y_in = torch.randn(4,128)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())
    #
    # x_in = x_in.to(device)  # 方法一：将input这个tensor转换成了CUDA 类型
    #
    #
    # y_in = y_in.to(device)
    # C = ContentEncoder(nf_cnt =64, n_downs=2, n_res=2, norm='in' ,act='relu', pad = 'reflect', use_sn=False)
    # C.to(device)
    #
    # re,_,_ = C(x_in)
    # print(re.shape)
    #
    #
    #
    # G = Generator()
    #
    #
    # import torch
    # import torchvision
    # from pytorch_model_summary import summary
    #
    # dummy_input = torch.randn(4, 3, 256, 256)
    # print(summary(G, dummy_input, show_input=False, show_hierarchical=False))
