import torch
import torch.nn as nn

# A block of 3d convolution/transpose convolution + batch normalization + relu/leaky relu (+ dropout)
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, act = "relu", use_dropout = False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 4, 2, 1, bias = False)
            if down
            else nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1, bias = False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels = 4, features = 64):
        super().__init__()
        
        # Encoding
        self.initial_down = nn.Sequential(
            nn.Conv3d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down = True, act = "leaky", use_dropout = False)
        self.down2 = Block(features * 2, features * 4, down = True, act = "leaky", use_dropout = False)
        self.down3 = Block(features * 4, features * 8, down = True, act = "leaky", use_dropout = False)
        self.down4 = Block(features * 8, features * 8, down = True, act = "leaky", use_dropout = False)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(),
        )
        
        # Decoding
        self.up1 = Block(features * 8, features * 8, down = False, act = "relu", use_dropout = True)
        self.up2 = Block(features * 8 * 2, features * 8, down = False, act = "relu", use_dropout = True)
        self.up3 = Block(features * 8 * 2, features * 4, down = False, act = "relu", use_dropout = False)
        self.up4 = Block(features * 4 * 2, features * 2, down = False, act = "relu", use_dropout = False)
        self.up5 = Block(features * 2 * 2, features, down = False, act = "relu", use_dropout = False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(features * 2, 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid(),
        )
        
        # Second arm for weight prediction
        self.weight1 = nn.Sequential( #128 --> 64
            nn.Conv3d(in_channels, 1, (3, 4, 4), (1, 2, 2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight2 = nn.Sequential( #64 --> 32
            nn.Conv3d(1, 1, (3, 4, 4), (1, 2, 2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight3 = nn.Sequential( #32 --> 16
            nn.Conv3d(1, 1, (3, 4, 4), (1, 2, 2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight4 = nn.Sequential( # 16 --> 8
            nn.Conv3d(1, 1, (3, 4, 4), (1, 2, 2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight5 = nn.Sequential( # 8 --> 4
            nn.Conv3d(1, 1, (3, 4, 4), (1, 2, 2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight6 = nn.Sequential( # 4 --> 2
            nn.Conv3d(1, 1, (3, 4, 4), (1, 2, 2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight7 = nn.Sequential( # 2 --> 1
            nn.Conv3d(1, 1, (3, 4, 4), (1, 2, 2), 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):

        weight = self.weight7(self.weight6(self.weight5(self.weight4(self.weight3(self.weight2(self.weight1(x)))))))
        
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        bottleneck = self.bottleneck(d5)
        up1 = self.up1(bottleneck)
        del bottleneck       
        up2 = self.up2(torch.cat([up1, d5], dim = 1))
        del d5
        del up1
        up3 = self.up3(torch.cat([up2, d4], dim = 1))       
        del d4
        del up2
        up4 = self.up4(torch.cat([up3, d3], dim = 1))
        del d3
        del up3        
        up5 = self.up5(torch.cat([up4, d2], dim = 1))
        del d2
        del up4
        #return self.final_up(torch.cat([up5, d1], dim = 1)), weight
        return self.final_up(torch.cat([up5, d1], dim = 1)) * weight


## TESTING SECTION ##

def test():
    x = torch.randn(1, 4, 192, 128, 128)
    model = Generator(in_channels = 4, features = 64)
    preds = model(x)
    print(preds.shape)

    
if __name__ == "__main__":
    test()