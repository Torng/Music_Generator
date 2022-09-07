import numpy as np
import torch
from mido import MidiFile, MidiTrack, Message


def denormalize(new_music):
    # control ,value,note,velocity,time,program_change,control_change,note_on,current_time
    new_music[0, :, 0:4] *= 127  # control
    new_music[0, :, 4] *= 5415  # control
    new_music[0, :, -1] *= 4095
    new_music[0, :, 0:4] = torch.where(new_music[0, :, 0:4] <= 0, 0, torch.floor(new_music[0, :, 0:4]))
    new_music[0, :, 0:4] = torch.where(new_music[0, :, 0:4] > 127, 127, torch.floor(new_music[0, :, 0:4]))
    new_music[0, :, 4] = torch.where(new_music[0, :, 4] <= 0, 0, torch.floor(new_music[0, :, 4]))
    new_music[0, :, 4] = torch.where(new_music[0, :, 4] > 5415, 5415, torch.floor(new_music[0, :, 4]))
    # new_music[0, :, -1] = torch.where(new_music[0, :, -1] <= 0, 0, torch.floor(new_music[0, :, -1]))
    # new_music[0, :, -1] = torch.where(new_music[0, :, -1] > 4095, 4095, torch.floor(new_music[0, :, -1]))


def to_midi(new_music):
    new_mid = MidiFile()
    track = MidiTrack()
    for i in range(new_music.shape[1]):
        channel = 0
        program = 0
        index = new_music[0, i, 5:8].max(-1)[1].view(1).item()
        if index == 0:
            time = int(new_music[0, i, 4].item())
            ms = Message(type='program_change', channel=channel, program=program, time=time)
        elif index == 1:
            control = int(new_music[0, i, 0].item())
            value = int(new_music[0, i, 1].item())
            time = int(new_music[0, i, 4].item())
            ms = Message(type='control_change', channel=channel, control=control, value=value, time=time)
        elif index == 2:
            note = int(new_music[0, i, 2].item())
            velocity = int(new_music[0, i, 3].item())
            time = int(new_music[0, i, 4].item())
            ms = Message(type='note_on', channel=channel, note=note, velocity=velocity, time=time)
        track.append(ms)
    new_mid.tracks.append(track)
    new_mid.save("new_midi.mid")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {}".format(device))
model = torch.load("model_set/model_100", map_location=device)
model.eval()

z = torch.randn(1, 128, 1, 1, device=device)
new_music = model.forward(z)

# new_music = new_music[0]
denormalize(new_music)
to_midi(new_music)
