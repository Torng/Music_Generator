from matplotlib import pyplot as plt
from preprocess import Preprocess
from Module.discriminator import Discriminator
from Module.generator import Generator
import torch
from torch import nn
import torch.optim as optim
from ray import tune
from torch.utils.data import DataLoader
from pathlib import Path
from midi_utils import denormalize, notes_to_midi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {}".format(device))
g_net = Generator().to(device)
d_net = Discriminator().to(device)

preprocess = Preprocess('maestro-v3.0.0/2011')

dl = DataLoader(preprocess.whole_training_data, batch_size=32, shuffle=True, drop_last=True)
print("Starting Training Loop...")


def gradient_penalty(critic, real_image, fake_image, device=None):
    batch_size, channel, width = real_image.shape
    fake_image = fake_image.to(device)
    # alpha is selected randomly between 0 and 1
    alpha = torch.rand(batch_size, 1, 1, device=device).repeat(1, channel, width)
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


@ray.remote(num_gpus=1)
def train(config):
    iterator = iter(dl)
    alpha = config['alpha'] if config else 10
    train_d_count = config['train_d'] if config else 5
    d_lr = config['d_lr'] if config else 0.00005
    g_lr = config['g_lr'] if config else 0.00005
    epochs = config['epochs'] if config else 1000
    optimizerD = optim.RMSprop(d_net.parameters(), lr=d_lr)
    optimizerG = optim.RMSprop(g_net.parameters(), lr=g_lr)
    loss_d_list = []
    loss_g_list = []
    for epoch in range(epochs):
        # For each batch in the dataloader
        for _ in range(train_d_count):
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
            z = torch.randn(b_size, 128, 1, device=device)
            xf = g_net(z)  # gradient would be passed down
            predf = d_net(xf)
            # min predf
            lossf = predf.mean()
            loss_D = lossf - lossr  # max
            gp = gradient_penalty(d_net, real_cpu, xf, device)
            loss_D = loss_D + gp * alpha
            optimizerD.zero_grad()
            loss_D.backward()
            optimizerD.step()
        z = torch.randn(b_size, 128, 1, device=device)
        xf = g_net(z)
        predf = d_net(xf)
        loss_G = -predf.mean()  # min
        # optimize
        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()
        loss_g_list.append(loss_G)
        loss_d_list.append(loss_D)

        if epoch % 2 == 0:
            print("epoch:{0} ==> lossDr:{1}, lossDf:{2}, lossD:{3},lossG:{4}".format(epoch, lossr, lossf, loss_D,
                                                                                     -loss_G))
        if epoch % 10 == 0:
            path = Path("model_set/")
            path.mkdir(exist_ok=True)
            output_path = path / ("model_" + str(epoch))
            torch.save(g_net, output_path)
        # if epoch % 100 == 0:
        #     path = Path("output_music/")
        #     path.mkdir(exist_ok=True)
        #     output_path = path / ("music_" + str(epoch) + ".midi")
        #     z = torch.randn(1, 128, 1, device=device)
        #     xf = g_net(z)
        #     music = denormalize(xf, preprocess.midi_std, preprocess.midi_mean)
        #     notes_to_midi(music, str(output_path), preprocess.instrument_name)
    return {'net': g_net, 'loss_d': loss_d_list[-1], 'loss_g': loss_g_list[-1]}