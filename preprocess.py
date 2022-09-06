import pandas as pd
import torchaudio
from pathlib import Path
import numpy as np
from mido import MidiFile, MidiTrack, Message
from collections import defaultdict


class Preprocess:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.whole_training_data = self.load_data()
        self.training_data = []

    def load_data(self):
        training_data = []
        for file_path in self.folder_path.glob("2004/*.midi"):
            midi = MidiFile(file_path)
            mete_data, midi_df = self.data_process(midi)
            if mete_data:
                training_data.append((mete_data, midi_df))
        max_time = max([data['time'].max() for meta_data, data in training_data])
        training_data = self.normalize(training_data, max_time)
        return training_data

    def data_process(self, data):
        types = ['program_change', 'control_change', 'note_on', 'end_of_track']
        columns = ['channel', 'program', 'control', 'value', 'note', 'velocity', 'time']
        result_df = pd.DataFrame(columns=columns)
        midi_tracks = data.tracks
        mete_data = midi_tracks[0]
        result_dic = defaultdict(list)
        current_count = 0
        if len(midi_tracks[1]) < 4096:
            return None, None
        for midi_track in midi_tracks[1]:
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
            midi_type = midi_track.type
            if midi_track.is_meta:
                continue
            if current_count == 4095:
                # result_dic['channel'].append(channel)
                # result_dic['program'].append(program)
                result_dic['control'].append(control)
                result_dic['value'].append(value)
                result_dic['note'].append(note)
                result_dic['velocity'].append(velocity)
                result_dic['time'].append(1)
                result_dic['program_change'].append(0)
                result_dic['control_change'].append(0)
                result_dic['note_on'].append(0)
                result_dic['end_of_track'].append(1)
                result_dic['current_time'].append(current_count)
                break
            if midi_track.type == 'program_change':
                channel = midi_track.channel
                program = midi_track.program
                time = midi_track.time
                program_change = 1
            elif midi_track.type == 'control_change':
                channel = midi_track.channel
                control = midi_track.control
                value = midi_track.value
                time = midi_track.time
                control_change = 1
            elif midi_track.type == 'note_on':
                channel = midi_track.channel
                note = midi_track.note
                velocity = midi_track.velocity
                time = midi_track.time
                note_on = 1
            elif midi_track.type == 'end_of_track':
                time = midi_track.time
                end_of_track = 1
            # result_dic['channel'].append(channel)
            # result_dic['program'].append(program)
            result_dic['control'].append(control)
            result_dic['value'].append(value)
            result_dic['note'].append(note)
            result_dic['velocity'].append(velocity)
            result_dic['time'].append(time)
            result_dic['current_time'].append(current_count)
            result_dic['program_change'].append(program_change)
            result_dic['control_change'].append(control_change)
            result_dic['note_on'].append(note_on)
            result_dic['end_of_track'].append(end_of_track)
            current_count += 1
        return mete_data, pd.DataFrame.from_dict(result_dic)

    def normalize(self, training_data, max_time):
        result_data = []
        for meta_data, data in training_data:
            # data['channel'] /= 15
            # data['program'] /= 127
            data['control'] /= 127
            data['value'] /= 127
            data['note'] /= 127
            data['time'] /= max_time
            data['velocity'] /= 127
            data['current_time'] /= 3837
            result_data.append(data.to_numpy().astype(np.float32))
        return result_data
