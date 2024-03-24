import torch
import math
import numpy
from Note import Note
from typing import List
from mido import MidiFile, MidiTrack, Message


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

    def in_seconds(self, ticks) -> float:
        return ticks / self.ticks_per_second

    def to_start_time_slower(self, discretization_step: int) -> torch.Tensor:
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
    

    def to_start_time_tensor(self, discretization_step: int) -> torch.Tensor:
        # Discretize song into time intervals    
        num_steps: int = math.ceil(self.song_length * discretization_step)
        midi_note_range = 128
        

        notes_tensor, interval_start_times, interval_end_times = self.prepare_arrays(discretization_step)

        start_times = notes_tensor[:, 0]
        end_times = notes_tensor[:, 1]
        interval_start_times = interval_start_times.unsqueeze(1)
        interval_end_times = interval_end_times.unsqueeze(1)
        active_notes = self.active_notes(interval_start_times, interval_end_times, start_times, end_times)

        discrete_frames = self.discrete_frames_calc(num_steps, notes_tensor, active_notes)
        
        return discrete_frames
    
    def discrete_frames_calc(self, num_steps, notes_tensor, active_notes):
        discrete_frames = torch.zeros(num_steps, 128)
        note_values = notes_tensor[:, 2]
        
        unique_note_values = torch.unique(note_values)
        # Get for each unique note value the indices of the notes with the same value in note_values tensor in vectorized / tensor form.
        note_indices = [torch.nonzero(note_values == note_value).flatten() for note_value in unique_note_values]

        # For each unique note value, sum the activations of the notes with the same value.
        # These two lines are the most time consuming part of the function. Maybe we could vectorize this part as well somehow.
        sum_notes = [torch.sum(active_notes[:, indices], dim=1) for indices in note_indices]
        for index, note_value in enumerate(unique_note_values):
            discrete_frames[:, note_value] = sum_notes[index]
        
        return discrete_frames

    def active_notes(self, interval_start_times, interval_end_times, start_times, end_times):
        active_notes = (start_times <= interval_start_times) & (end_times >= interval_end_times)
        return active_notes


    def prepare_arrays(self, discretization_step: int):
        num_steps: int = math.ceil(self.song_length * discretization_step)

        notes_tensor = torch.stack([torch.tensor([n.start_time, n.end_time, n.value]) for n in self.notes])
        interval_start_times = torch.arange(0, num_steps) * self.ticks_per_second / discretization_step
        interval_end_times = torch.arange(1, num_steps + 1) * self.ticks_per_second / discretization_step

        return notes_tensor, interval_start_times, interval_end_times

    
    def start_time_tensor_to_midi(tensor: torch.Tensor, out_path, discretization_step: int, bpm = 120, tempo = 500000, note_threshold = 0.5):
        default_velocity = 100

        tensor = tensor.to('cpu')

        # Add one zero tensor to the end of the given tensor to make sure all notes are lifted at the end.
        tensor = torch.cat((tensor, torch.zeros(1, tensor.shape[1])), 0)

        # Create a new MIDI file (Type 0)
        song = MidiFile(type=0)
        # meta_track = MidiTrack()
        # meta_track.append(Message('set_tempo', tempo=tempo, time=0))
        # meta_track.append(Message('end_of_track', time=1))
        # song.tracks.append(meta_track)

        ticks_per_frame = tempo / discretization_step

        # tempo = tempo
        # tempo_bpm = 60000000 / tempo
        # microseconds_per_tick = tempo / ticks_per_beat
        # ticks_per_second = 1000000 / microseconds_per_tick

        track = MidiTrack()

        last_notes = torch.zeros(128, dtype=torch.bool)
        last_tick = 0

        for step in range(tensor.shape[0]):
            if step % 40000 == 0:
                print(f"Processing step {step} of {tensor.shape[0]}")

            # current_second = step / discretization_step
            # current_micro_second = current_second * 1000000
            # current_tick = round(current_micro_second / tempo) * 120
            # current_tick = int(step * ticks_per_frame)
            bpm = 120 * 4
            current_tick = int(step * ticks_per_frame / bpm)
            
            frame = tensor[step]
            # Assume any note value above 0.5 is a note
            # pressed_notes = set((frame > 0.5).nonzero().flatten())
            pressed_notes = frame > note_threshold

            # Newly pressed notes.
            new_notes = (pressed_notes & ~last_notes).nonzero().flatten()
            start_notes = [Message('note_on', note=int(note_value), velocity=default_velocity, time=0) for note_value in new_notes]
            
            # Released notes.
            released_notes = (~pressed_notes & last_notes).nonzero().flatten()
            end_notes = [Message('note_on', note=int(note_value), velocity=0, time=0) for note_value in released_notes]

            if len(start_notes) > 0:
                start_notes[0].time = current_tick - last_tick            
                last_tick = current_tick
            elif len(end_notes) > 0:
                end_notes[0].time = current_tick - last_tick
                last_tick = current_tick

            for note in start_notes:
                track.append(note)
            for note in end_notes:
                track.append(note)

            last_notes = pressed_notes
            
        # track.append(Message('end_of_track', time=1))
        song.tracks.append(track)

        # Save the MIDI file
        song.save(out_path)            
            