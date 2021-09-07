import torch
import torch.nn as nn
import torch.nn.functional as F

#based on pix2pix
class Seg_Net(nn.Module):
    def __init__(self
                 ):
        super(Seg_Net, self).__init__()
        self.conv1_1 = nn.Conv3d(4, 18, 3, stride = 1, padding = 1, bias = False)
        self.conv1_2 = nn.Conv3d(18, 18, 3, stride = 1, padding = 1, bias = False)
        self.conv1_3 = nn.Conv3d(18, 18, 3, stride = 1, padding = 1, bias = False)
        self.conv2_1 = nn.Conv3d(18, 36, 3, stride = 1, padding = 1, bias = False)
        self.conv2_2 = nn.Conv3d(36, 36, 3, stride = 1, padding = 1, bias = False)
        self.conv2_3 = nn.Conv3d(36, 36, 3, stride = 1, padding = 1, bias = False)
        self.conv3_1 = nn.Conv3d(36, 54, 3, stride = 1, padding = 1, bias = False)
        self.conv3_2 = nn.Conv3d(54, 54, 3, stride = 1, padding = 1, bias = False)
        self.conv3_3 = nn.Conv3d(54, 54, 3, stride = 1, padding = 1, bias = False)
        self.conv4_1 = nn.Conv3d(54, 72, 3, stride = 1, padding = 1, bias = False)
        self.conv4_2 = nn.Conv3d(72, 72, 3, stride = 1, padding = 1, bias = False)
        self.conv4_3 = nn.Conv3d(72, 72, 3, stride = 1, padding = 1, bias = False)
        self.conv5_1 = nn.Conv3d(72, 90, 3, stride = 1, padding = 1, bias = False)
        self.conv5_2 = nn.Conv3d(90, 90, 3, stride = 1, padding = 1, bias = False)
        self.conv5_3 = nn.Conv3d(90, 90, 3, stride = 1, padding = 1, bias = False)
        self.conv6_1 = nn.Conv3d(90, 108, 3, stride = 1, padding = 1, bias = False)
        self.conv6_2 = nn.Conv3d(108, 108, 3, stride = 1, padding = 1, bias = False)
        self.conv6_3 = nn.Conv3d(108, 108, 3, stride = 1, padding = 1, bias = False)
        self.conv6_4 = nn.Conv3d(108, 108, 3, stride = 1, padding = 1, bias = False)
        self.pool = nn.MaxPool3d(2, 2, return_indices=False)
        self.convup1 = nn.Conv3d(180, 90, 3, stride = 1, padding=1)
        self.convup2 = nn.Conv3d(144, 72, 3, stride = 1, padding=1)
        self.convup3 = nn.Conv3d(108, 54, 3, stride = 1, padding=1)
        self.convup4 = nn.Conv3d(72, 36, 3, stride = 1, padding=1)
        self.convup5 = nn.Conv3d(36, 18, 3, stride = 1, padding=1)
        self.uptrans1 = nn.ConvTranspose3d(108, 90, 2, stride=2, padding = (1,0,0), output_padding = (1,0,0))
        self.uptrans2 = nn.ConvTranspose3d(90, 72, 2, stride=2)
        self.uptrans3 = nn.ConvTranspose3d(72, 54, 2, stride=2)
        self.uptrans4 = nn.ConvTranspose3d(54, 36, 2, stride=2, padding = (1,0,0), output_padding = (1,0,0))
        self.uptrans5 = nn.ConvTranspose3d(36, 18, 2, stride=2)
        self.BN1_1 = nn.BatchNorm3d(18)
        self.BN1_2 = nn.BatchNorm3d(18)
        self.BN1_3 = nn.BatchNorm3d(18)
        self.BN1_4 = nn.BatchNorm3d(18)
        self.BN1_up = nn.BatchNorm3d(18)
        self.BN2_1 = nn.BatchNorm3d(36)
        self.BN2_2 = nn.BatchNorm3d(36)
        self.BN2_3 = nn.BatchNorm3d(36)
        self.BN2_4 = nn.BatchNorm3d(36)
        self.BN2_up = nn.BatchNorm3d(36)
        self.BN3_1 = nn.BatchNorm3d(54)
        self.BN3_2 = nn.BatchNorm3d(54)
        self.BN3_3 = nn.BatchNorm3d(54)
        self.BN3_4 = nn.BatchNorm3d(54)
        self.BN3_up = nn.BatchNorm3d(54)
        self.BN4_1 = nn.BatchNorm3d(72)
        self.BN4_2 = nn.BatchNorm3d(72)
        self.BN4_3 = nn.BatchNorm3d(72)
        self.BN4_4 = nn.BatchNorm3d(72)
        self.BN4_up = nn.BatchNorm3d(72)
        self.BN5_1 = nn.BatchNorm3d(90)
        self.BN5_2 = nn.BatchNorm3d(90)
        self.BN5_3 = nn.BatchNorm3d(90)
        self.BN5_4 = nn.BatchNorm3d(90)
        self.BN5_up = nn.BatchNorm3d(90)
        self.BN6_1 = nn.BatchNorm3d(108)
        self.BN6_2 = nn.BatchNorm3d(108)
        self.BN6_3 = nn.BatchNorm3d(108)
        self.BN6_4 = nn.BatchNorm3d(108)
        self.output = nn.Conv3d(18, 1, 3, stride = 1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()
        self.paddeven = nn.ConstantPad3d((0,0,0,0,0,1),0)
        
    def forward(self, x):

        x1 = self.BN1_2(self.leakyrelu(self.conv1_2(self.BN1_1(self.leakyrelu(self.conv1_1(x))))))
        x = self.pool(x1)

        x2 = self.BN2_2(self.leakyrelu(self.conv2_2(self.BN2_1(self.leakyrelu(self.conv2_1(x))))))
        x = self.pool(self.paddeven(x2))

        x3 = self.BN3_2(self.leakyrelu(self.conv3_2(self.BN3_1(self.leakyrelu(self.conv3_1(x))))))
        x = self.pool(x3)

        x4 = self.BN4_2(self.leakyrelu(self.conv4_2(self.BN4_1(self.leakyrelu(self.conv4_1(x))))))
        x = self.pool(x4)

        x5 = self.BN5_2(self.leakyrelu(self.conv5_2(self.BN5_1(self.leakyrelu(self.conv5_1(x))))))
        x = self.pool(self.paddeven(x5))
        
        x = self.BN6_2(self.leakyrelu(self.conv6_2(self.BN6_1(self.leakyrelu(self.conv6_1(x))))))
        x = self.BN6_4(self.leakyrelu(self.conv6_4(self.BN6_3(self.leakyrelu(self.conv6_3(x))))))
        
        x = self.BN5_up(self.leakyrelu(self.uptrans1(x)))
        x = torch.cat((x,x5), dim=1)
        del x5
        x = self.BN5_4(self.leakyrelu(self.conv5_3(self.BN5_3(self.leakyrelu(self.convup1(x))))))

        x = self.BN4_up(self.leakyrelu(self.uptrans2(x)))

        x = torch.cat((x, x4), dim=1)
        del x4
        x = self.BN4_4(self.leakyrelu(self.conv4_3(self.BN4_3(self.leakyrelu(self.convup2(x))))))

        x = self.BN3_up(self.leakyrelu(self.uptrans3(x)))
        x = torch.cat((x, x3), dim=1)
        del x3
        x = self.BN3_4(self.leakyrelu(self.conv3_3(self.BN3_3(self.leakyrelu(self.convup3(x))))))

        x = self.BN2_up(self.leakyrelu(self.uptrans4(x)))
        x = torch.cat((x, x2), dim=1)
        del x2
        x = self.BN2_4(self.leakyrelu(self.conv2_3(self.BN2_3(self.leakyrelu(self.convup4(x))))))##

        x = self.BN1_up(self.leakyrelu(self.uptrans5(x)))
        x = torch.cat((x, x1), dim=1)
        del x1
        x = self.BN1_4(self.leakyrelu(self.conv1_3(self.BN1_3(self.leakyrelu(self.convup5(x))))))
        x = self.sig(self.output(x))
        return x

# A block of 3d convolution or transpose convolution, batch normalization and relu/leakyrelu, used in pixnet
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act ="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 4, 2, 1, bias = False)
            if down
            else nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1, bias = False),
            #nn.GroupNorm(int(out_channels/8),out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

#pix2pix
class pixnet(nn.Module):
    def __init__(self, in_channels=4, features = 64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv3d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
        ) #192,128,128 -> 96,64,64 & 4 -> 64 channels
        
        self.down1 = Block(features, features*2, down=True, act = "leaky", use_dropout=True) # 96x64x64 --> 48x32x32  64->128 channels
        self.down2 = Block(features*2, features*4, down=True, act = "leaky", use_dropout=True) # 48x32x32 --> 24x16x16   128 -> 256 channels
        self.down3 = Block(features*4, features*8, down=True, act = "leaky", use_dropout=True) # 24x16x16 --> 12x8x8    256 -> 512 channels
        self.down4 = Block(features*8, features*8, down=True, act = "leaky", use_dropout=True) # 12x8x8 --> 6x4x4  512-> 512 channels
        self.bottleneck = nn.Sequential(
            nn.Conv3d(features*8, features*8, 4, 2, 1), nn.ReLU(), #6x4x4 -> 3x2x2  512-> 512 channels
        )
        self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout=True) #3x2x2 -> 6x4x4 512->512 channels
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)#6x4x4 -> 12x8x8 512+512 -> 512 channels
        self.up3 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=True)#12x8x8 -> 24x16x16 512+512 -> 256 channels
        self.up4 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=True)#24x16x16->48x32x32 256+256 -> 128 channels
        self.up5 = Block(features*2*2, features, down=False, act="relu", use_dropout=True)#48x32x32 -> 96x64x64 128+128 -> 64 channels
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(features*2, 1,kernel_size=4,stride=2,padding=1), #96x64x64 -> 192x128x128 64+64 -> 1 channels
            nn.Sigmoid(),
        )

        self.weight1 = nn.Sequential(#128 --> 64
            nn.Conv3d(in_channels, 1, (3,4,4), (1,2,2), 1),
            nn.LeakyReLU(0.2),
        )
        self.weight2 = nn.Sequential(#64 -->32
            nn.Conv3d(1,1, (3,4,4), (1,2,2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight3 = nn.Sequential(#32 --> 16
            nn.Conv3d(1,1, (3,4,4), (1,2,2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight4 = nn.Sequential(#16-->8
            nn.Conv3d(1,1, (3,4,4), (1,2,2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight5 = nn.Sequential(#8 -->4
            nn.Conv3d(1,1, (3,4,4), (1,2,2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight6 = nn.Sequential(#4 --> 2 
            nn.Conv3d(1,1, (3,4,4), (1,2,2), 1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(0.2),
        )
        self.weight7 = nn.Sequential(#2-->1
            nn.Conv3d(1,1, (3,4,4), (1,2,2), 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        #self.FClayer1 = nn.Sequential(
        #    nn.Linear(6144, 192),
        #    nn.ReLU(),
        #)
        #self.FClayer2 = nn.Sequential(
        #    nn.Linear(192,192),
        #    nn.ReLU(),
        #)
        #self.FClayer3 = nn.Sequential(
        #    nn.Linear(192,192),
        #    nn.Sigmoid(),
        #)
        #self.fin_weight = nn.Parameter(torch.ones(192,1,1)*0.014)
        #self.fin_weight.requires_grad = True
        
    def forward(self, x):
        d1 = self.initial_down(x)
        w1 = self.weight7(self.weight6(self.weight5(self.weight4(self.weight3(self.weight2(self.weight1(x)))))))
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        bottleneck = self.bottleneck(d5)
        
        #w1 = torch.unsqueeze(torch.unsqueeze(self.FClayer3(self.FClayer2(self.FClayer1(torch.flatten(bottleneck)))),1),2)
        
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1,d5], dim=1))
        u3 = self.up3(torch.cat([u2,d4], dim=1))
        u4 = self.up4(torch.cat([u3,d3], dim=1))
        u5 = self.up5(torch.cat([u4,d2], dim=1))
        u5 = self.final_up(torch.cat([u5,d1], dim=1))
        #u5 = torch.mul(u5, w1)
        return u5, w1

def test():
    x = torch.randn((1,4,192,128,128))
    model = pixnet()
    preds = model(x)
    #weight = model.fin_weight
    #print(weight)
    #print(preds.shape)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)

if __name__ == "__main__":
    test()        