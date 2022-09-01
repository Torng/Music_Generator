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
# new_music = new_music.detach().numpy()
# new_music = np.int16(new_music/np.max(np.abs(new_music)) * 32767)
# write('test.wav', 44100, new_music)
torchaudio.save("test.wav", new_music, 44100)