import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_t_1 = nn.Conv1d(6, 64, kernel_size=4, stride=(2))
        self.conv_t_2 = nn.Conv1d(64, 128, kernel_size=4, stride=(2))
        self.conv_t_3 = nn.Conv1d(128, 256, kernel_size=4, stride=(2))
        self.conv_t_4 = nn.Conv1d(256, 1, kernel_size=4, stride=(2))
        # self.conv_t_5 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=2)
        # self.conv_t_6 = nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=2)
        self.b_n_1 = nn.BatchNorm1d(64)
        self.b_n_2 = nn.BatchNorm1d(128)
        self.b_n_3 = nn.BatchNorm1d(256)
        # self.b_n_5 = nn.BatchNorm1d(128)
        # self.fc_1 = nn.Linear(512, 256)
        # self.fc_2 = nn.Linear(256, 1)
        # self.fc_2 = nn.Linear(512, 128)
        # self.fc_3 = nn.Linear(128, 32)
        # self.conv_t_2 = nn.Conv1d(1, 1, kernel_size=4)
        # self.main = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )

    def forward(self, x):
        x = self.conv_t_1(x)
        x = self.b_n_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv_t_2(x)
        x = self.b_n_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv_t_3(x)
        x = self.b_n_3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv_t_4(x)
        # x = self.b_n_4(x)
        x = F.leaky_relu(x, 0.2)
        x = x.mean(dim=(-2, -1))
        # x = x.view(1, -1)
        # x = self.fc_1(x)
        # x = self.fc_2(x)
        # x = self.fc_3(x)
        x = F.sigmoid(x)
        return x
