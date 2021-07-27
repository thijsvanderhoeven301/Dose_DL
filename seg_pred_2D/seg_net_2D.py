import torch
import torch.nn as nn
import torch.nn.functional as F

#pix2pix model
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act ="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias = False,padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class pixnet(nn.Module):
    def __init__(self, in_channels=4, features = 64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ) #64,64 -> 32,32 & 4 -> 64 channels
        
        self.down1 = Block(features, features*2, down=True, act = "leaky", use_dropout=False) # 32x32 --> 16x16  64->128 channels
        self.down2 = Block(features*2, features*4, down=True, act = "leaky", use_dropout=False) # 16x16 --> 8x8   128 -> 256 channels
        self.down3 = Block(features*4, features*8, down=True, act = "leaky", use_dropout=False) # 8x8 --> 4x4    256 -> 512 channels
        self.down4 = Block(features*8, features*8, down=True, act = "leaky", use_dropout=False) # 4x4 --> 2x2  512-> 512 channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode ="reflect"), nn.ReLU(), #2x2 -> 1x1  512-> 512 channels
        )
        self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout=True) #1x1 -> 2x2 512->512 channels
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)# 2x2 -> 14x4 512+512 -> 512 channels
        self.up3 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=False)#4x4 -> 8x8 512+512 -> 256 channels
        self.up4 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=False)#8x8->16x16 256+256 -> 128 channels
        self.up5 = Block(features*2*2, features, down=False, act="relu", use_dropout=False)#16x16 -> 32x32 128+128 -> 64 channels
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, 1 ,kernel_size=4,stride=2,padding=1), #32x32 -> 64x64 64+64 -> 1 channel
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        bottleneck = self.bottleneck(d5)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1,d5], dim=1))
        u3 = self.up3(torch.cat([u2,d4], dim=1))
        u4 = self.up4(torch.cat([u3,d3], dim=1))
        u5 = self.up5(torch.cat([u4,d2], dim=1))
        return self.final_up(torch.cat([u5,d1],dim=1))