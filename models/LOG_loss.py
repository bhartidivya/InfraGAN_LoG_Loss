#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np


# In[3]:


class LogLoss(nn.Module):
    print("into  LogLoss")
    
    def __init__(self, use_gpu = True):
        super(LogLoss, self).__init__()
        self.log_block = LogEachBlock()
        self.loss = nn.L1Loss()
        if use_gpu:
            self.log_block.cuda()

    def __call__(self, input_A, input_B):
        log_A = self.log_block(input_A)
#         plt.subplot(1,2,1)
#         plt.title("log A")
#         plt.imshow(log_A.detach().cpu().numpy()[0,0,:,:],cmap='gray')
        
#         print("logA",log_A)
        
        log_B = self.log_block(input_B)
#         plt.subplot(1,2,2)
#         plt.title("log B")
#         plt.imshow(log_B.detach().cpu().numpy()[0,0,:,:],cmap='gray')
        
#         print("logb",log_B)
        return self.loss(log_A, log_B)


class LogBlock(nn.Module):
    """LoG Filter Block for LoG loss"""
#     print("Log Block")
    def __init__(self):
        super(LogBlock, self).__init__()

        ## RGB to Gray Block
        np_filter1 = np.array([[0.2989, 0.5870, 0.1140]]);
        conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(np_filter1).float().unsqueeze(2).unsqueeze(2))

        ## LoG Filter Block
        
        np_filter_2 =np.array([[0, -1, 0] ,[-1 ,4 ,-1] ,[0 ,-1 ,0]])
        
        conv2 =nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight =nn.Parameter(torch.from_numpy(np_filter_2).float().unsqueeze(0).unsqueeze(0))

        self.main = nn.Sequential(conv1, conv2)

        ## Fix all weights
        for param in self.main.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.main(x)


class LogEachBlock(nn.Module):
    """LoG Filter Block for LoG Loss, applied for each color channel"""
    print("Log each block")
    def __init__(self):
        super(LogEachBlock, self).__init__()

        ## LoG Filter Block
        np_filter2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight = nn.Parameter(torch.from_numpy(np_filter2).float().unsqueeze(0).unsqueeze(0))

        self.avg_pool = torch.nn.AvgPool3d((3, 1, 1), stride=1)
        self.smooth = torch.nn.AvgPool2d(3, stride=1, padding=1)

        for param in self.conv2.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Split each channel
        chan1 = x[:, 0, :, :]
        chan2 = x[:, 1, :, :]
        chan3 = x[:, 2, :, :]

        # Match dimension batch x chan x height x width
        chan1 = chan1.unsqueeze(1)
        chan2 = chan2.unsqueeze(1)
        chan3 = chan3.unsqueeze(1)

        # chan1 = self.smooth(chan1)
        # chan2 = self.smooth(chan2)
        # chan3 = self.smooth(chan3)

        filt1 = self.conv2(chan1)
        filt2 = self.conv2(chan2)
        filt3 = self.conv2(chan3)
#         plt.subplot(1,1,1)
#         plt.imshow(filt1.detach().cpu().numpy()[0,0,:,:],cmap='gray')
        # Concat channels and apply pooling
        concat = torch.cat((filt1, filt2, filt3), 1)
        output = self.avg_pool(concat)
#         print("output",output.shape)
        return output


# In[ ]:




