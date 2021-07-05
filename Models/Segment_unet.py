import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


# Define neural net
class custom_linear(nn.Module):
    def __init__(self, size):
        super().__init__()
        weights = torch.ones(size)
        self.weights = nn.Parameter(weights)

        nn.init.uniform_(self.weights)

    def forward(self, x):
        w_x = self.weights[None, None, :, None, None] * x
        return w_x


class SegNet3D(nn.Module):
    def __init__(self,
                 ):
        super(SegNet3D, self).__init__()
        self.input = nn.Conv3d(4, 8, 3, padding=1)
        self.conv1_2 = nn.Conv3d(8, 8, 3, padding=1)
        self.conv1_3 = nn.Conv3d(8, 8, 3, padding=1)
        self.conv2_1 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv2_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv2_3 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv3_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3_3 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv4_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv4_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv4_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv5_1 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv5_2 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv5_3 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv5_4 = nn.Conv3d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool3d(2, 2, return_indices=False)
        self.convup1 = nn.Conv3d(128, 64, 3, padding=1)
        self.convup2 = nn.Conv3d(64, 32, 3, padding=1)
        self.convup3 = nn.Conv3d(32, 16, 3, padding=1)
        self.convup4 = nn.Conv3d(16, 8, 3, padding=1)
        self.uptrans1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.uptrans2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.uptrans3 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.uptrans4 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.output = nn.Conv3d(8, 1, 1, padding=0)
        self.dice_output = nn.Conv3d(8, 2, 1, padding=0)
        self.GN1_1 = nn.GroupNorm(1, 8)
        self.GN1_2 = nn.GroupNorm(1, 8)
        self.GN1_3 = nn.GroupNorm(1, 8)
        self.GN1_4 = nn.GroupNorm(1, 8)
        self.GN1_up = nn.GroupNorm(1, 8)
        self.GN2_1 = nn.GroupNorm(2, 16)
        self.GN2_2 = nn.GroupNorm(2, 16)
        self.GN2_3 = nn.GroupNorm(2, 16)
        self.GN2_4 = nn.GroupNorm(2, 16)
        self.GN2_up = nn.GroupNorm(2, 16)
        self.GN3_1 = nn.GroupNorm(4, 32)
        self.GN3_2 = nn.GroupNorm(4, 32)
        self.GN3_3 = nn.GroupNorm(4, 32)
        self.GN3_4 = nn.GroupNorm(4, 32)
        self.GN3_up = nn.GroupNorm(4, 32)
        self.GN4_1 = nn.GroupNorm(8, 64)
        self.GN4_2 = nn.GroupNorm(8, 64)
        self.GN4_3 = nn.GroupNorm(8, 64)
        self.GN4_4 = nn.GroupNorm(8, 64)
        self.GN4_up = nn.GroupNorm(8, 64)
        self.GN5_1 = nn.GroupNorm(16, 128)
        self.GN5_2 = nn.GroupNorm(16, 128)
        self.GN5_3 = nn.GroupNorm(16, 128)
        self.GN5_4 = nn.GroupNorm(16, 128)
        self.Sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(1, 1, bias=False)
        #self.clinear = custom_linear(144)


    def forward(self, x):
        x1 = F.relu(self.GN1_2(self.conv1_2(F.relu(self.GN1_1(self.input(x))))))
        x = self.pool(x1)

        x2 = F.relu(self.GN2_2(self.conv2_2(F.relu(self.GN2_1(self.conv2_1(x))))))
        x = self.pool(x2)

        x3 = F.relu(self.GN3_2(self.conv3_2(F.relu(self.GN3_1(self.conv3_1(x))))))
        x = self.pool(x3)

        x4 = F.relu(self.GN4_2(self.conv4_2(F.relu(self.GN4_1(self.conv4_1(x))))))
        x = self.pool(x4)

        x = F.relu(self.GN5_2(self.conv5_2(F.relu(self.GN5_1(self.conv5_1(x))))))
        x = F.relu(self.GN5_4(self.conv5_4(F.relu(self.GN5_3(self.conv5_3(x))))))

        x = F.relu(self.GN4_up(self.uptrans1(x))) # When upsampling with only transpose convolution
        
        x = torch.cat((x, x4), dim=1)
        del x4
        x = F.relu(self.GN4_4(self.conv4_3(F.relu(self.GN4_3(self.convup1(x))))))

        x = F.relu(self.GN3_up(self.uptrans2(x))) # When upsampling with only transpose convolution
        
        x = torch.cat((x, x3), dim=1)
        del x3
        x = F.relu(self.GN3_4(self.conv3_3(F.relu(self.GN3_3(self.convup2(x))))))

        x = F.relu(self.GN2_up(self.uptrans3(x))) # When upsampling with only transpose convolution
        
        x = torch.cat((x, x2), dim=1)
        del x2
        x = F.relu(self.GN2_3(self.conv2_3(F.relu(self.GN2_3(self.convup3(x))))))

        x = F.relu(self.GN1_up(self.uptrans4(x))) # When upsampling with only transpose convolution
        
        x = torch.cat((x, x1), dim=1)
        del x1
        x = F.relu(self.GN1_4(self.conv1_3(F.relu(self.GN1_3(self.convup4(x))))))
        #x = self.softmax(self.dice_output(x))
        #x = self.clinear(self.Sig(self.output(x)))#
        x = F.relu(self.output(x))
        #x = self.output(x)
        return x


class SegNet2D(nn.Module):
    def __init__(self,
                 ):
        super(SegNet2D, self).__init__()
        self.input = nn.Conv3d(4, 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv1_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv1_2 = nn.Conv2d(140, 256, 3, padding=1)
        self.conv2_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv2_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv3_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv3_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv4_1 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.conv4_2 = nn.Conv2d(2048, 2048, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.convup1 = nn.Conv2d(2048, 1024, 3, padding=1)
        self.convup2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.convup3 = nn.Conv2d(512, 256, 3, padding=1)
        self.uptrans1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.uptrans2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.uptrans3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.output = nn.Conv2d(256, 140, 1, padding=0)
        self.GN1 = nn.GroupNorm(16, 256)
        self.GN2 = nn.GroupNorm(32, 512)
        self.GN3 = nn.GroupNorm(64, 1024)
        self.GN4 = nn.GroupNorm(128, 2048)
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        x1 = F.relu(self.GN1(self.conv1_2(torch.squeeze(F.relu(self.input(x)), dim=0))))
        x, ind1 = self.pool(x1)

        x2 = F.relu(self.GN2(self.conv2_2(F.relu(self.GN2(self.conv2_1(x))))))
        x, ind2 = self.pool(x2)

        x = F.relu(self.GN3(self.conv3_2(F.relu(self.GN3(self.conv3_1(x))))))
        x = F.relu(self.GN3(self.conv3_2(F.relu(self.GN3(self.conv3_2(x))))))

        x = F.relu(self.GN2(self.uptrans2(x)))  # When upsampling with only transpose convolution
        del ind2
        x = torch.cat((x, x2), dim=1)
        del x2
        x = F.relu(self.GN2(self.conv2_2(F.relu(self.GN2(self.convup2(x))))))

        x = F.relu(self.GN1(self.uptrans3(x)))  # When upsampling with only transpose convolution
        del ind1
        x = torch.cat((x, x1), dim=1)
        del x1
        x = F.relu(self.GN1(self.conv1_1(F.relu(self.GN1(self.convup3(x))))))
        x = self.Sig(self.output(x)) #
        #x = self.output(x)
        return x