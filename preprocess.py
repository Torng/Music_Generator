from pathlib import Path
import numpy as np
import torch
from midi_utils import midi_to_notes

class Preprocess:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.whole_training_data = self.load_data()
        self.training_data = []

    def load_data(self):
        training_data = []
        for file_path in self.folder_path.glob("*.midi"):
            midi = midi_to_notes(str(file_path))
            # midi_array = self.preprocess_midi(midi)
            tensor_midi = torch.tensor(midi.iloc[:128].to_numpy().astype(np.float32)).T
            training_data.append(tensor_midi)
        return training_data




