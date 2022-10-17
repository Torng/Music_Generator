from collections import defaultdict
from typing import Optional

import pandas as pd
import pretty_midi
from matplotlib import pyplot as plt
import numpy as np


def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100,  # note loudness
                  is_drum=False) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name), is_drum=is_drum)

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=int(note['velocity']),
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    is_drum = False
    if instrument.is_drum:
        is_drum = True
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

    notes = defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        if (note.pitch >= 28 and note.pitch <= 54) or not is_drum:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['velocity'].append(note.velocity)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()}), instrument_name, is_drum


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)


def denormalize(t, midi_std, midi_mean):
    t = t.T
    t[:, 0] = t[:, 0] * midi_std['pitch'] + midi_mean['pitch']
    t[:, 1] = t[:, 1] * midi_std['velocity'] + midi_mean['velocity']
    t[:, 2] = t[:, 2] * midi_std['start'] + midi_mean['start']
    t[:, 3] = t[:, 3] * midi_std['end'] + midi_mean['end']
    t[:, 4] = t[:, 4] * midi_std['step'] + midi_mean['step']
    t[:, 5] = t[:, 5] * midi_std['duration'] + midi_mean['duration']
    return pd.DataFrame(np.array(t.resize(128,6).detach()))
