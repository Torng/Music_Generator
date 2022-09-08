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
num_epochs = 500
g_net = Generator().to(device)
d_net = Discriminator().to(device)
# Initialize BCELoss function
criterion = nn.BCELoss()
# g_net(torch.randn(1, 128, 1, 1))
# d_net(torch.randn(1, 1, 256, 24))
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
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dl, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        d_net.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = d_net(real_cpu).view(-1)
        if i % 32 == 0:
            errD_real = criterion(output, label)
            errD_real.backward()
        else:
            errD_real = torch.tensor(0)
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 128, 1, 1, device=device)
        # Generate fake image batch with G
        fake = g_net(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = d_net(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        if i % 32 == 0:
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        g_net.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = d_net(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 16 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, 16,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(preprocess.training_data) - 1)):
            with torch.no_grad():
                fake = g_net(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    if epoch % 10 == 0:
        path = Path("model_set/")
        path.mkdir(exist_ok=True)
        output_path = path / ("model_" + str(epoch))
        torch.save(g_net, output_path)

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
