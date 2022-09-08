import numpy as np
import torch
from mido import MidiFile, MidiTrack, Message


def denormalize(new_music):
    # control ,value,note,velocity,time,program_change,control_change,note_on,current_time
    new_music[1] *= 127
    new_music[1] = torch.where(new_music[1] <= 0, 0, torch.floor(new_music[1]))
    new_music[1] = torch.where(new_music[1] > 127, 127, torch.floor(new_music[1]))
    new_music[0] = torch.where(new_music[1] <= 0, 0, torch.round(new_music[0]))
    new_music[0] = torch.where(new_music[1] > 1, 1, torch.round(new_music[0]))


def to_midi(new_music):
    new_mid = MidiFile()
    track = MidiTrack()
    pre_time = 0
    playing_notes = []
    for current_beat in range(new_music.shape[1]):
        is_note = (new_music[0, current_beat] == 1).nonzero(as_tuple=True)[0]
        not_note = (new_music[0, current_beat] == 0).nonzero(as_tuple=True)[0]
        # pre_time =current_beat
        for not_note_idx in not_note:
            not_note_num = not_note_idx.item()
            if not_note_num in playing_notes:
                current_time = current_beat * 60
                ms = Message(type='note_on', channel=9, note=not_note_num + 31, velocity=0,
                             time=current_time - pre_time)
                track.append(ms)
                playing_notes.remove(not_note_num)
                pre_time = current_time
        for note_idx in is_note:
            if new_music[0, current_beat, note_idx] == 1:
                note_idx_num = note_idx.item()
                velocity = int(new_music[1, current_beat, note_idx].item())
                ms = Message(type='note_on', channel=9, note=note_idx_num + 31, velocity=velocity,
                             time=current_beat * 60 - pre_time)
                track.append(ms)
                playing_notes.append(note_idx_num)
                pre_time = current_beat * 60
    new_mid.tracks.append(track)
    new_mid.save("new_midi.mid")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {}".format(device))
model = torch.load("model_set/model_100", map_location=device)
model.eval()

z = torch.randn(1, 128, 1, 1, device=device)
new_music = model.forward(z)
new_music = new_music[0]
# new_music = new_music[0]
denormalize(new_music)
to_midi(new_music)
