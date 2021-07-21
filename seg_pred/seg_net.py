import torch
import torch.nn as nn

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
        x = self.output(x)
        return x
        