import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_t_1 = nn.Conv1d(1, 1, kernel_size=4, stride=2)
        self.conv_t_2 = nn.Conv1d(1, 1, kernel_size=4, stride=4)
        self.b_n_1 = nn.BatchNorm1d(1)
        self.fc = nn.Linear(496, 128)
        self.fc_2 = nn.Linear(128, 4)
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
        x = F.relu(x)
        x = self.conv_t_2(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        x = self.conv_t_2(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        x = self.conv_t_2(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        x = self.conv_t_2(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        x = self.conv_t_2(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        x = self.conv_t_1(x)
        x = self.b_n_1(x)
        x = x.view(-1)
        x = self.fc(x)
        x = self.fc_2(x)
        x = F.sigmoid(x)
        return x
