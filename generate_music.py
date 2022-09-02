import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {}".format(device))
model = torch.load("model_1", map_location=device)
model.eval()

z = torch.randn(1, 1, 128, device=device)
new_music = model.forward(z)
new_music = new_music.view(1, -1)
torchaudio.save("test.wav", new_music, 44100)