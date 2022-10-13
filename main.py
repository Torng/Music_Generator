from matplotlib import pyplot as plt
from preprocess import Preprocess
from Module.discriminator import Discriminator
from Module.generator import Generator
import torch
from torch import nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {}".format(device))

# preprocess.set_batch_size(32)
num_epochs = 10000
g_net = Generator().to(device)
d_net = Discriminator().to(device)
# Initialize BCELoss function
criterion = nn.BCELoss()
# g_net(torch.randn(1, 128, 1, 1))
# d_net(torch.randn(10, 1, 1024, 24))
preprocess = Preprocess('funk_music')

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, 128, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
lr = 0.0002
beta1 = 0.5
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(d_net.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(g_net.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
dl = DataLoader(preprocess.whole_training_data, batch_size=32, shuffle=True, drop_last=True)
print("Starting Training Loop...")
# For each epoch
save_name = 1


def gradient_penalty(critic, real_image, fake_image, device=None):
    batch_size, channel, height, width = real_image.shape
    fake_image = fake_image.to(device)
    # alpha is selected randomly between 0 and 1
    alpha = torch.rand(batch_size, 1, 1, 1, device=device).repeat(1, channel, height, width)
    # interpolated image=randomly weighted average between a real and fake image
    # interpolated image ← alpha *real image  + (1 − alpha) fake image
    interpolatted_image = (alpha * real_image) + (1 - alpha) * fake_image
    interpolatted_image = interpolatted_image.to(device)
    # calculate the critic score on the interpolated image
    interpolated_score = critic(interpolatted_image)

    # take the gradient of the score wrt to the interpolated image
    gradient = torch.autograd.grad(inputs=interpolatted_image,
                                   outputs=interpolated_score,
                                   retain_graph=True,
                                   create_graph=True,
                                   grad_outputs=torch.ones_like(interpolated_score, device=device)
                                   )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


iterator = iter(dl)
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for _ in range(5):
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(dl)
        # data = next(iterator)
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        predr = d_net(real_cpu).view(-1)
        # maximize predr, therefore minus sign
        lossr = predr.mean()
        z = torch.randn(b_size, 128, 1, 1, device=device)
        xf = g_net(z)  # gradient would be passed down
        predf = d_net(xf)
        # min predf
        lossf = predf.mean()
        loss_D = lossf - lossr  # max
        gp = gradient_penalty(d_net, real_cpu, xf, device)
        loss_D = loss_D + gp * 10
        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()
    z = torch.randn(b_size, 128, 1, 1, device=device)
    xf = g_net(z)
    predf = d_net(xf)
    loss_G = -predf.mean()  # min
    # optimize
    optimizerG.zero_grad()
    loss_G.backward()
    optimizerG.step()

    if epoch % 2 == 0:
        print("epoch:{0} ==> lossDr:{1}, lossDf:{2}, lossD:{3},lossG:{4}".format(epoch, lossr, lossf, -loss_D, loss_G))
    if epoch % 10 == 0:
        path = Path("model_set/")
        path.mkdir(exist_ok=True)
        output_path = path / ("model_" + str(epoch))
        torch.save(g_net, output_path)
