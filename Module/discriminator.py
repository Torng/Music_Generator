import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_t_1 = nn.Conv1d(6, 64, kernel_size=4, stride=2)
        self.conv_t_2 = nn.Conv1d(64, 128, kernel_size=4, stride=2)
        self.conv_t_3 = nn.Conv1d(128, 256, kernel_size=4, stride=2)
        self.conv_t_4 = nn.Conv1d(256, 1, kernel_size=4, stride=2)
        self.b_n_1 = nn.BatchNorm1d(64)
        self.b_n_2 = nn.BatchNorm1d(128)
        self.b_n_3 = nn.BatchNorm1d(256)
        self.fc = nn.Linear(6,1)

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
        x = self.fc(x)
        # x = x.mean(dim=(-2, -1))
        # x = x.view(1, -1)
        # x = self.fc_1(x)
        # x = self.fc_2(x)
        # x = self.fc_3(x)
        x = F.sigmoid(x)
        return x
