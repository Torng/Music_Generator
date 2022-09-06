import numpy as np
import torch
from mido import MidiFile, MidiTrack, Message


def denormalize(new_music):
    # control ,value,note,velocity,time,program_change,control_change,note_on,end_of_track,current_time
    new_music[0, :, 0:5] *= 127  # control
    new_music[0, :, -1] *= 4095
    new_music[0, :, 0:5] = torch.where(new_music[0, :, 0:5] <= 0, 0, torch.floor(new_music[0, :, 0:5]))
    new_music[0, :, 0:5] = torch.where(new_music[0, :, 0:5] > 127, 127, torch.floor(new_music[0, :, 0:5]))
    new_music[0, :, 5] = torch.where(new_music[0, :, 5] <= 0, 0, torch.floor(new_music[0, :, 5]))
    new_music[0, :, 5] = torch.where(new_music[0, :, 5] > 4095, 4095, torch.floor(new_music[0, :, 5]))


def to_midi(new_music):
    new_mid = MidiFile()
    track = MidiTrack()
    for i in range(new_music.shape[1]):
        channel = 0
        program = 0
        control = 0
        value = 0
        note = 0
        velocity = 0
        time = 0
        program_change = 0
        control_change = 0
        note_on = 0
        end_of_track = 0
        index = new_music[0, i, 5:9].max(-1)[1].view(1).item()
        if index == 0:
            time = new_music[0, i, 4].item()
            ms = Message(type='program_change', channel=channel, program=program, time=time)
        elif index == 1:
            control = new_music[0, i, 0].item()
            value = new_music[0, i, 1].item()
            time = new_music[0, i, 4].item()
            ms = Message(type='control_change', channel=channel, control=control, value=value, time=time)
        elif index == 2:
            note = new_music[0, i, 2].item()
            velocity = new_music[0, i, 3].item()
            time = new_music[0, i, 4].item()
            ms = Message(type='note_on', channel=channel, note=note, velocity=velocity, time=time)
        track.append(ms)
    new_mid.tracks.append(track)
    new_mid.save("new_midi_mid")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {}".format(device))
model = torch.load("model_set/model_100", map_location=device)
model.eval()

z = torch.randn(1, 128, 1, 1, device=device)
new_music = model.forward(z)

# new_music = new_music[0]
denormalize(new_music)
to_midi(new_music)
