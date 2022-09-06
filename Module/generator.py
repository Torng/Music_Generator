# Generator Code
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_t_1 = nn.ConvTranspose2d(128, 1024, kernel_size=4, stride=(4, 1), bias=False)
        self.b_n_1 = nn.BatchNorm2d(1024)
        self.conv_t_2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=(4, 1), padding=(0, 0), bias=False)
        self.b_n_2 = nn.BatchNorm2d(512)
        self.conv_t_3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=(4, 1), padding=(0, 2), bias=False)
        self.b_n_3 = nn.BatchNorm2d(256)
        self.conv_t_4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=(4, 1), padding=(0, 2), bias=False)
        self.b_n_4 = nn.BatchNorm2d(128)
        self.conv_t_5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=(4, 1), padding=(0, 1), bias=False)
        self.b_n_5 = nn.BatchNorm2d(64)
        self.conv_t_6 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=(4, 1), bias=False)
        self.b_n_6 = nn.BatchNorm2d(1)

    def get_conv_t_output_size(self, input_size, kernel_size, stride, padding, output_padding=0):
        return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding

    def forward(self, x):
        x = self.conv_t_1(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        x = self.conv_t_2(x)
        x = self.b_n_2(x)
        x = F.relu(x)
        x = self.conv_t_3(x)
        x = self.b_n_3(x)
        x = F.relu(x)
        x = self.conv_t_4(x)
        x = self.b_n_4(x)
        x = F.relu(x)
        x = self.conv_t_5(x)
        x = self.b_n_5(x)
        x = F.relu(x)
        x = self.conv_t_6(x)
        x = self.b_n_6(x)
        x = F.tanh(x)
        x = x.view(-1, 4096, 9)
        return x
