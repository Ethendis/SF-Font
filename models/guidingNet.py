from torch import nn
import torch.nn.functional as F
import torch
try:
    from models.blocks import Conv2dBlock, FRN
except:
    from blocks import Conv2dBlock, FRN
# from dynamic_conv import Dynamic_conv2d
import random

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg19cut': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'N'],
}



class GuidingNet(nn.Module):
    def __init__(self, img_size=64, output_k={'cont': 128, 'disc': 10}):
        super(GuidingNet, self).__init__()
        #network layers setting
        self.linear1 = nn.Linear(8*128, 4*128)
        self.linear2 = nn.Linear(4*128, 2*128)
        self.linear3 = nn.Linear(2*128, 128)


        self.features = make_layers(cfg['vgg11'], True)
        self.disc = nn.Linear(512, output_k['disc'])
        self.cont = nn.Linear(512, output_k['cont'])

        self._initialize_weights()

    def forward(self, x, sty=False):
        # x = x.view(1, 256 * 2048 * 3)
        # x = self.linear1(x)
        # x = self.linear2(x)
        # x = self.linear3(x)
        # x = x.view(1, 3, 256, 256)
        # print(x.shape)
        # exit()
        x = self.features(x)
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
    def split(self,x):
        y = torch.split(x, 128, dim=3)  # 在第4维度分离
        #y = torch.split(x, 128, dim=3)  # 在第4维度分离
        sty_feat = []
        for x in y:
            x = self.features(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            flat = x.view(x.size(0), -1)

            disc = self.cont(flat)
            #print(disc.shape)
            sty_feat.append(disc)
        s_ref = random.choice(sty_feat)#
        #s_ref = torch.cat(sty_feat, dim=1)

        # print(s_ref.shape)
        # exit()
        # s_ref = self.linear1(s_ref)
        # s_ref = self.linear2(s_ref)
        # s_ref = self.linear3(s_ref)
        return s_ref



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


if __name__ == '__main__':
    import torch
    C = GuidingNet(64)
    x_in = torch.randn(4, 3, 256, 256)
    #x_in = torch.randn(4, 3, 128, 128)
    sty = C.moco(x_in)
    cls = C.iic(x_in)
    #print(sty.shape, cls.shape)
    x_inn =torch.randn(1,3,256,256*8)
    #x_inn = torch.randn(1, 3, 128, 128 * 8)
    s = C.split(x_inn)
    #print(s.shape)
    import torch
    import torchvision
    from pytorch_model_summary import summary
    #print(C)
    #dummy_input = torch.randn(1, 3, 256 ,256)
    #print(summary(C, dummy_input, show_input=False, show_hierarchical=False))

