import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


# Define neural net
class UNet(nn.Module):
    def __init__(self,
                 ):
        super(UNet, self).__init__()
        self.input = nn.Conv3d(5, 16, 3, padding=1)
        self.conv1_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv1_3 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv2_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv2_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv2_3 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv3_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv4_1 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv4_2 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv4_3 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv5_1 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv5_2 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv5_3 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv5_4 = nn.Conv3d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool3d(2, 2, return_indices=False)
        self.unpool = nn.MaxUnpool3d(2, 2)
        self.convup1 = nn.Conv3d(256, 128, 3, padding=1)
        self.convup2 = nn.Conv3d(128, 64, 3, padding=1)
        self.convup3 = nn.Conv3d(64, 32, 3, padding=1)
        self.convup4 = nn.Conv3d(32, 16, 3, padding=1)
        self.uptrans1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.uptrans2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.uptrans3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.uptrans4 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.output = nn.Conv3d(16, 1, 3, padding=1)
        self.GN1_1 = nn.GroupNorm(2, 16)
        self.GN1_2 = nn.GroupNorm(2, 16)
        self.GN1_3 = nn.GroupNorm(2, 16)
        self.GN1_4 = nn.GroupNorm(2, 16)
        self.GN1_up = nn.GroupNorm(2, 16)
        self.GN2_1 = nn.GroupNorm(4, 32)
        self.GN2_2 = nn.GroupNorm(4, 32)
        self.GN2_3 = nn.GroupNorm(4, 32)
        self.GN2_4 = nn.GroupNorm(4, 32)
        self.GN2_up = nn.GroupNorm(4, 32)
        self.GN3_1 = nn.GroupNorm(8, 64)
        self.GN3_2 = nn.GroupNorm(8, 64)
        self.GN3_3 = nn.GroupNorm(8, 64)
        self.GN3_4 = nn.GroupNorm(8, 64)
        self.GN3_up = nn.GroupNorm(8, 64)
        self.GN4_1 = nn.GroupNorm(16, 128)
        self.GN4_2 = nn.GroupNorm(16, 128)
        self.GN4_3 = nn.GroupNorm(16, 128)
        self.GN4_4 = nn.GroupNorm(16, 128)
        self.GN4_up = nn.GroupNorm(16, 128)
        self.GN5_1 = nn.GroupNorm(32, 256)
        self.GN5_2 = nn.GroupNorm(32, 256)
        self.GN5_3 = nn.GroupNorm(32, 256)
        self.GN5_4 = nn.GroupNorm(32, 256)

    def forward(self, x):
        x1 = self.GN1_2(F.relu(self.conv1_2(self.GN1_1(F.relu(self.input(x))))))
        x = self.pool(x1)

        x2 = self.GN2_2(F.relu(self.conv2_2(self.GN2_1(F.relu(self.conv2_1(x))))))
        x = self.pool(x2)

        x3 = self.GN3_2(F.relu(self.conv3_2(self.GN3_1(F.relu(self.conv3_1(x))))))
        x = self.pool(x3)

        x4 = self.GN4_2(F.relu(self.conv4_2(self.GN4_1(F.relu(self.conv4_1(x))))))
        x = self.pool(x4)

        x = self.GN5_2(F.relu(self.conv5_2(self.GN5_1(F.relu(self.conv5_1(x))))))
        x = self.GN5_4(F.relu(self.conv5_4(self.GN5_3(F.relu(self.conv5_3(x))))))

        x = self.GN4_up(F.relu(self.uptrans1(x))) # When upsampling with only transpose convolution
        
        x = torch.cat((x, x4), dim=1)
        del x4
        x = self.GN4_4(F.relu(self.conv4_3(self.GN4_3(F.relu(self.convup1(x))))))

        x = self.GN3_up(F.relu(self.uptrans2(x))) # When upsampling with only transpose convolution
        
        x = torch.cat((x, x3), dim=1)
        del x3
        x = self.GN3_4(F.relu(self.conv3_3(self.GN3_3(F.relu(self.convup2(x))))))

        x = self.GN2_up(F.relu(self.uptrans3(x))) # When upsampling with only transpose convolution
        
        x = torch.cat((x, x2), dim=1)
        del x2
        x = self.GN2_4(F.relu(self.conv2_3(self.GN2_3(F.relu(self.convup3(x))))))

        x = self.GN1_up(F.relu(self.uptrans4(x))) # When upsampling with only transpose convolution
        
        x = torch.cat((x, x1), dim=1)
        del x1
        x = self.GN1_4(F.relu(self.conv1_3(self.GN1_3(F.relu(self.convup4(x))))))
        x = F.relu(self.output(x))
        return x


class SeqUNet(nn.Module):
    def __init__(self,
                 ):
        super(SeqUNet, self).__init__()
        self.input = nn.Conv3d(6, 16, 3, padding=1)
        self.conv1_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv1_3 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv2_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv2_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv2_3 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv3_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv4_1 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv4_2 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv4_3 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv5_1 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv5_2 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv5_3 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv5_4 = nn.Conv3d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool3d(2, 2, return_indices=False)
        self.convup1 = nn.Conv3d(256, 128, 3, padding=1)
        self.convup2 = nn.Conv3d(128, 64, 3, padding=1)
        self.convup3 = nn.Conv3d(64, 32, 3, padding=1)
        self.convup4 = nn.Conv3d(32, 16, 3, padding=1)
        self.uptrans1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.uptrans2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.uptrans3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.uptrans4 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.output = nn.Conv3d(16, 1, 3, padding=1)
        self.GN1_1 = nn.GroupNorm(2, 16)
        self.GN1_2 = nn.GroupNorm(2, 16)
        self.GN1_3 = nn.GroupNorm(2, 16)
        self.GN1_4 = nn.GroupNorm(2, 16)
        self.GN1_up = nn.GroupNorm(2, 16)
        self.GN2_1 = nn.GroupNorm(4, 32)
        self.GN2_2 = nn.GroupNorm(4, 32)
        self.GN2_3 = nn.GroupNorm(4, 32)
        self.GN2_4 = nn.GroupNorm(4, 32)
        self.GN2_up = nn.GroupNorm(4, 32)
        self.GN3_1 = nn.GroupNorm(8, 64)
        self.GN3_2 = nn.GroupNorm(8, 64)
        self.GN3_3 = nn.GroupNorm(8, 64)
        self.GN3_4 = nn.GroupNorm(8, 64)
        self.GN3_up = nn.GroupNorm(8, 64)
        self.GN4_1 = nn.GroupNorm(16, 128)
        self.GN4_2 = nn.GroupNorm(16, 128)
        self.GN4_3 = nn.GroupNorm(16, 128)
        self.GN4_4 = nn.GroupNorm(16, 128)
        self.GN4_up = nn.GroupNorm(16, 128)
        self.GN5_1 = nn.GroupNorm(32, 256)
        self.GN5_2 = nn.GroupNorm(32, 256)
        self.GN5_3 = nn.GroupNorm(32, 256)
        self.GN5_4 = nn.GroupNorm(32, 256)

    def forward(self, x):
        x1 = self.GN1_2(F.relu(self.conv1_2(self.GN1_1(F.relu(self.input(x))))))
        x = self.pool(x1)

        x2 = self.GN2_2(F.relu(self.conv2_2(self.GN2_1(F.relu(self.conv2_1(x))))))
        x = self.pool(x2)

        x3 = self.GN3_2(F.relu(self.conv3_2(self.GN3_1(F.relu(self.conv3_1(x))))))
        x = self.pool(x3)

        x4 = self.GN4_2(F.relu(self.conv4_2(self.GN4_1(F.relu(self.conv4_1(x))))))
        x = self.pool(x4)

        x = self.GN5_2(F.relu(self.conv5_2(self.GN5_1(F.relu(self.conv5_1(x))))))
        x = self.GN5_4(F.relu(self.conv5_4(self.GN5_3(F.relu(self.conv5_3(x))))))

        x = self.GN4_up(F.relu(self.uptrans1(x)))
        x = torch.cat((x, x4), dim=1)
        del x4
        x = self.GN4_4(F.relu(self.conv4_3(self.GN4_3(F.relu(self.convup1(x))))))

        x = self.GN3_up(F.relu(self.uptrans2(x)))
        x = torch.cat((x, x3), dim=1)
        del x3
        x = self.GN3_4(F.relu(self.conv3_3(self.GN3_3(F.relu(self.convup2(x))))))

        x = self.GN2_up(F.relu(self.uptrans3(x)))
        x = torch.cat((x, x2), dim=1)
        del x2
        x = self.GN2_4(F.relu(self.conv2_3(self.GN2_3(F.relu(self.convup3(x))))))

        x = self.GN1_up(F.relu(self.uptrans4(x)))
        x = torch.cat((x, x1), dim=1)
        del x1
        x = self.GN1_4(F.relu(self.conv1_3(self.GN1_3(F.relu(self.convup4(x))))))
        x = F.relu(self.output(x))
        return x


class InDoseUNet(nn.Module):
    def __init__(self,
                 ):
        super(InDoseUNet, self).__init__()
        self.input = nn.Conv3d(1, 16, 3, padding=1)
        self.conv1_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv1_3 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv2_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv2_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv2_3 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv3_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv4_1 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv4_2 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv4_3 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv5_1 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv5_2 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv5_3 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv5_4 = nn.Conv3d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool3d(2, 2, return_indices=False)
        self.convup1 = nn.Conv3d(256, 128, 3, padding=1)
        self.convup2 = nn.Conv3d(128, 64, 3, padding=1)
        self.convup3 = nn.Conv3d(64, 32, 3, padding=1)
        self.convup4 = nn.Conv3d(32, 16, 3, padding=1)
        self.uptrans1 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.uptrans2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.uptrans3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.uptrans4 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.output = nn.Conv3d(16, 1, 3, padding=1)
        self.GN1_1 = nn.GroupNorm(2, 16)
        self.GN1_2 = nn.GroupNorm(2, 16)
        self.GN1_3 = nn.GroupNorm(2, 16)
        self.GN1_4 = nn.GroupNorm(2, 16)
        self.GN1_up = nn.GroupNorm(2, 16)
        self.GN2_1 = nn.GroupNorm(4, 32)
        self.GN2_2 = nn.GroupNorm(4, 32)
        self.GN2_3 = nn.GroupNorm(4, 32)
        self.GN2_4 = nn.GroupNorm(4, 32)
        self.GN2_up = nn.GroupNorm(4, 32)
        self.GN3_1 = nn.GroupNorm(8, 64)
        self.GN3_2 = nn.GroupNorm(8, 64)
        self.GN3_3 = nn.GroupNorm(8, 64)
        self.GN3_4 = nn.GroupNorm(8, 64)
        self.GN3_up = nn.GroupNorm(8, 64)
        self.GN4_1 = nn.GroupNorm(16, 128)
        self.GN4_2 = nn.GroupNorm(16, 128)
        self.GN4_3 = nn.GroupNorm(16, 128)
        self.GN4_4 = nn.GroupNorm(16, 128)
        self.GN4_up = nn.GroupNorm(16, 128)
        self.GN5_1 = nn.GroupNorm(32, 256)
        self.GN5_2 = nn.GroupNorm(32, 256)
        self.GN5_3 = nn.GroupNorm(32, 256)
        self.GN5_4 = nn.GroupNorm(32, 256)

    def forward(self, x):
        x1 = self.GN1_2(F.relu(self.conv1_2(self.GN1_1(F.relu(self.input(x))))))
        x = self.pool(x1)

        x2 = self.GN2_2(F.relu(self.conv2_2(self.GN2_1(F.relu(self.conv2_1(x))))))
        x = self.pool(x2)

        x3 = self.GN3_2(F.relu(self.conv3_2(self.GN3_1(F.relu(self.conv3_1(x))))))
        x = self.pool(x3)

        x4 = self.GN4_2(F.relu(self.conv4_2(self.GN4_1(F.relu(self.conv4_1(x))))))
        x = self.pool(x4)

        x = self.GN5_2(F.relu(self.conv5_2(self.GN5_1(F.relu(self.conv5_1(x))))))
        x = self.GN5_4(F.relu(self.conv5_4(self.GN5_3(F.relu(self.conv5_3(x))))))

        x = self.GN4_up(F.relu(self.uptrans1(x)))
        x = torch.cat((x, x4), dim=1)
        del x4
        x = self.GN4_4(F.relu(self.conv4_3(self.GN4_3(F.relu(self.convup1(x))))))

        x = self.GN3_up(F.relu(self.uptrans2(x)))
        x = torch.cat((x, x3), dim=1)
        del x3
        x = self.GN3_4(F.relu(self.conv3_3(self.GN3_3(F.relu(self.convup2(x))))))

        x = self.GN2_up(F.relu(self.uptrans3(x)))
        x = torch.cat((x, x2), dim=1)
        del x2
        x = self.GN2_4(F.relu(self.conv2_3(self.GN2_3(F.relu(self.convup3(x))))))

        x = self.GN1_up(F.relu(self.uptrans4(x)))
        x = torch.cat((x, x1), dim=1)
        del x1
        x = self.GN1_4(F.relu(self.conv1_3(self.GN1_3(F.relu(self.convup4(x))))))
        x = F.relu(self.output(x))
        return x
