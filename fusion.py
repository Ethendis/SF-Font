import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import scipy.io as io
import math
import numpy as np


import sys
sys.path.append('..')
from modules import modulated_deform_conv
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


        query_matrix = self.linears_query(content_feats_reshape)



        residual = query_matrix
        query_matrix = query_matrix.view(batch, h * w, self.num_heads, d_channel)  # [B, HW, num_heads, C/num_heads]

        query_matrix = query_matrix.permute(0, 2, 3, 1)  # [B,num_heads,C/num_head,HW

        content_skeleton_permute = y.permute(0, 2, 3, 1)  # B,H,W,C
        batch, h, w, channel = content_skeleton_permute.shape

        content_skeleton_reshape = torch.reshape(content_skeleton_permute, (batch, h * w, channel))  # B, HW, C
        query_matrix_skeleton = self.linears_value(content_skeleton_reshape)
        query_matrix_skeleton = query_matrix_skeleton.view(batch, h * w, self.num_heads, d_channel)

        key_matrix = query_matrix_skeleton.permute(0, 2, 3, 1)  #
        v_matrix = query_matrix_skeleton.permute(0, 2, 1, 3)

       
        attention_mask = torch.matmul(query_matrix.permute(0, 1, 3, 2), key_matrix)  # [B, num_heads, HW, HW]
       
        attention_mask = attention_mask.permute(0, 1, 3, 2) / math.sqrt(
            h * w) 
        attention_mask = F.softmax(attention_mask, dim=-1)
        
        value_mask = torch.matmul(attention_mask, v_matrix)  # [B, num_heads, HW, C/num_heads]
        
        value_mask = value_mask.permute(0, 1, 3, 2)  # [B, num_heads, C/num_heads, HW]
        value_mask = torch.reshape(value_mask, (batch, channel, -1))

        value_mask = value_mask.view(batch, h * w, self.num_channels)  # [B, HW, C]

        value_mask = self.fc(value_mask)
        

        value_mask = self.layer_norm(value_mask)
        value_mask = value_mask.view(batch, self.num_channels, h * w)
        # print(value_mask.shape)
        feat_scs = value_mask.view(batch, self.num_channels, h, w)
        # print(feat_scs.shape)
        return feat_scs
if __name__  =="__main__":
    A = Multi_Head_Attention()
    x_in = torch.randn(1,256,64,64)
    y_in = torch.randn(1, 256, 64, 64)
    re = A(x_in,y_in)
    print(re.shape)