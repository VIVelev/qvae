# pylint: disable=no-member

import torch
import torch.nn as nn

__all__ = [
    'ResBlock',
    'ResStack',

    'ResEncoder',
    'ResDecoder',
]


class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_hiddens):
        super(ResBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens,
                kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, out_channels,
                kernel_size=1, stride=1, bias=False),
            nn.ReLU(True)
        )

    def forward(self, x):
        return x + self.residual(x)

class ResStack(nn.Module):

    def __init__(self, in_channels, out_channels, num_hiddens, num_res_layers):
        super(ResStack, self).__init__()

        self.num_res_layers = num_res_layers
        self.res_layers = nn.ModuleList([
            ResBlock(in_channels, out_channels, num_hiddens) for _ in range(num_res_layers)
        ])

    def forward(self, x):
        for layer in self.res_layers:
            x = layer(x)

        return x
    

class ResEncoder(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_res_hiddens, num_res_layers):
        super(ResEncoder, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels, num_hiddens//2, 
                        kernel_size=4, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(num_hiddens//2, num_hiddens,
                        kernel_size=4, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(num_hiddens, num_hiddens,
                        kernel_size=3, stride=1, padding=1)
        
        self.res_stack = ResStack(num_hiddens, num_hiddens, num_res_hiddens, num_res_layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        return self.res_stack(x)

class ResDecoder(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_res_hiddens, num_res_layers, rgb=True):
        super(ResDecoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels, num_hiddens,
                        kernel_size=3, stride=1, padding=1)
        
        self.res_stack = ResStack(num_hiddens, num_hiddens, num_res_hiddens, num_res_layers)
        
        self.conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens//2,
                        kernel_size=4, stride=2, padding=1)
        
        self.conv_trans_2 = nn.ConvTranspose2d(num_hiddens//2, 3 if rgb else 1,
                        kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.res_stack(x)

        x = self.conv_trans_1(x)
        x = torch.relu(x)

        x = self.conv_trans_2(x)
        return torch.sigmoid(x)
