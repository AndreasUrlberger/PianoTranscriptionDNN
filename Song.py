import torch
import math
import numpy
from Note import Note
from typing import List



class Song:
    def __init__(self, notes: List[Note], song_length: float, ticks_per_beat: int, tempo: int):
        self.notes =notes.copy()
        self.notes.sort(key=lambda x: x.start_time)
        self.song_length = song_length
        # Also call PPQ (pulses per quarter note)
        self.ticks_per_beat = ticks_per_beat
        # Tempo (in microseconds per quarter note) (usually at beginning of midi as meta message 'set_tempo')
        self.tempo = tempo
        self.tempo_bpm = 60000000 / tempo
        self.microseconds_per_tick = tempo / ticks_per_beat
        self.ticks_per_second = 1000000 / self.microseconds_per_tick

    def __repr__(self):
        return f"Song(Length: '{self.song_length:.2f}s', tempo: {self.tempo}, bpm: {self.tempo_bpm}, ticks_per_second: {self.ticks_per_second}, microsec_per_tick: {self.microseconds_per_tick}, notes: {self.notes})"


    def to_start_time_tensor(self, discretization_step: int) -> torch.Tensor:
        # Discretize song into time intervals    
        num_steps: int = math.ceil(self.song_length * discretization_step)
        num_notes = len(self.notes)
        midi_note_range = 128
        discrete_frames = torch.zeros(num_steps, midi_note_range)
        open_notes = {}
        
        # Notes are sorted by start time, so we do not have to iterate through all notes for each step
        note_index = 0
        for step in range(num_steps):
            # Calculate interval start and end in ticks.
            interval_start = float(step) * self.ticks_per_second / discretization_step
            interval_end = float(step + 1) * self.ticks_per_second / discretization_step

            # Check notes starting or stopping in current interval.
            frame = discrete_frames[step]
            contained_note = True
            while note_index < num_notes and contained_note:
                note = self.notes[note_index]

                if note.start_time < interval_end:
                    frame[note.value] = 1
                    if note.end_time > interval_end:
                        open_notes[note.value] = note
                    note_index += 1
                elif note.end_time < interval_end:
                    frame[note.value] = 1
                    note_index += 1
                else:
                    contained_note = False

            # Check notes that are still open from previous intervals  
            open_note_keys = set(open_notes.keys())
            for open_note_value in open_note_keys:
                open_note = open_notes[open_note_value]
                frame[open_note_value] = 1
                if open_note.end_time < interval_end:
                    del open_notes[open_note_value]
                    
        return discrete_frames
            
    def in_seconds(self, ticks) -> float:
        return ticks / self.ticks_per_second
            