# Generator Code
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_t_1 = nn.ConvTranspose1d(1, 1, kernel_size=4, stride=4, padding=0, bias=False)
        self.b_n_1 = nn.BatchNorm1d(1)
        self.conv_t_2 = nn.ConvTranspose1d(1, 1, kernel_size=4, stride=4, padding=0, bias=False, output_padding=3)
        self.conv_t_3 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=0, bias=False)

    def get_conv_t_output_size(self, input_size, kernel_size, stride, padding, output_padding=0):
        return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding

    def forward(self, x):
        x = self.conv_t_2(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        x = self.conv_t_1(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        x = self.conv_t_1(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        x = self.conv_t_3(x)
        x = self.b_n_1(x)
        x = F.relu(x)
        # x = self.conv_t_1(x)
        # x = self.b_n_1(x)
        # x = F.relu(x)
        x = self.conv_t_1(x)
        x = self.b_n_1(x)
        x = F.tanh(x)
        return x
