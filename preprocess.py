from pathlib import Path
import numpy as np
import torch
from midi_utils import midi_to_notes


class Preprocess:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.training_data = []
        self.midi_std = {}
        self.midi_mean = {}
        self.instrument_name = ""
        self.is_drum = False
        self.whole_training_data = self.load_data()

    def load_data(self):
        training_data = []
        for file_path in self.folder_path.glob("*.midi"):
            midi, instrument_name, is_drum = midi_to_notes(str(file_path))
            midi = midi.iloc[:128]
            self.is_drum = is_drum
            self.midi_std = midi.std()
            self.midi_mean = midi.mean()
            midi = (midi - midi.mean()) / (midi.std() + 1e-10)
            tensor_midi = torch.tensor(midi.to_numpy().astype(np.float32)).T
            training_data.append(tensor_midi)
        self.instrument_name = instrument_name
        return training_data
