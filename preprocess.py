from pathlib import Path
import numpy as np
from mido import MidiFile, MidiTrack, Message


class Preprocess:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.whole_training_data = self.load_data()
        self.training_data = []

    def load_data(self):
        training_data = []
        for file_path in self.folder_path.glob("*.midi"):
            midi = MidiFile(file_path)
            midi_array = self.preprocess_midi(midi)
            training_data.append(midi_array)
        return training_data

    def preprocess_midi(self, mid, per_beat=15):
        current_time = 0
        note_dic = {}
        current_midi = np.zeros((1, 1024, 24))
        for track in mid.tracks[1]:
            if type(track) == Message:
                if track.time + current_time > 15360:
                    break
                if track.type == 'note_on' and 54 >= track.note >= 31:
                    if track.note not in note_dic:
                        note_dic[track.note] = (current_time + track.time, track.velocity)
                    else:
                        start_index = round(note_dic[track.note][0] / per_beat)
                        # velocity = note_dic[track.note][1]
                        end_index = round((current_time + track.time) / per_beat)
                        current_midi[0, start_index:end_index + 1, track.note - 31] = 1
                        # current_midi[1, start_index:end_index + 1, track.note - 31] = velocity
                        del note_dic[track.note]
                current_time += track.time
        # current_midi[1] = current_midi[1] / 127
        current_midi = current_midi.astype(np.float32)
        return current_midi
