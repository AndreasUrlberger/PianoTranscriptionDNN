"""Data utility functions."""
import os

import torch
import csv

import torchaudio
import Utils as utils
import math
    
from mido import MidiFile
from Note import Note
from Song import Song

class MidiDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, split, discretization=100):
        self.root_dir_name = os.path.dirname(dataset_dir)
        self.discretization = discretization
        file_name = (split + '.txt').lower()
        paths_file = os.path.join(dataset_dir, file_name)

        with open(paths_file) as f:
            lines = f.read().splitlines()
            self.audio_files = lines[::2]
            self.midi_files = lines[1::2]

    def __getitem__(self, key):
        audio_file = self.audio_files[key]
        midi_file = self.midi_files[key]
        audio_tensor, _ = torchaudio.load(audio_file) # Don't need the sample rate
        # Average the channels to mono
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=False)
        parsed_midi_file = MidiFile(midi_file)
        notes = Note.midi_to_notes(parsed_midi_file)

        tempo = 500000
        for msg in parsed_midi_file.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

        song = Song(notes, parsed_midi_file.length, parsed_midi_file.ticks_per_beat, tempo)
        midi_tensor = song.to_start_time_tensor(discretization_step=self.discretization)
        # TODO This might create problems down the line with different audio lengths
        discrete_steps = midi_tensor.shape[0]

        waveform_length = audio_tensor.shape[0]
        waveform_length_per_midi_step = waveform_length / discrete_steps
        waveform_length_per_midi_step_int = int(waveform_length_per_midi_step)

        # Somewhat fast
        chunk_start_indices = [math.floor(i * waveform_length_per_midi_step) for i in range(discrete_steps)]      
        chunked_audio_tensor = torch.stack([audio_tensor[i:(i+waveform_length_per_midi_step_int)] for i in chunk_start_indices])
        chunked_audio_tensor = chunked_audio_tensor.reshape((discrete_steps, waveform_length_per_midi_step_int))

        # Slow        
        # chunked_audio_tensor = torch.zeros((discrete_steps, waveform_length_per_midi_step_int))
        # for i in range(discrete_steps):
        #     start = math.floor(i * waveform_length_per_midi_step)
        #     end = start + waveform_length_per_midi_step_int
        #     chunked_audio_tensor[i] = audio_tensor[start:end]

        # Slower 
        # hunk_start_indices = [math.floor(i * waveform_length_per_midi_step) for i in range(discrete_steps)]      
        # copy_indices = np.array([[i + j for j in range(waveform_length_per_midi_step_int)] for i in chunk_start_indices])
        # chunked_audio_tensor = audio_tensor[copy_indices]

        return chunked_audio_tensor, midi_tensor
    
    def get_audio_path(self, key):
        return self.audio_files[key]
    
    def get_midi_path(self, key):
        return self.midi_files[key]


    def __len__(self):
        # Probably unnecessary
        return len(min(self.audio_files, self.midi_files))
    
    def create_dataset_files(dataset_dir, output_dir):
        with open(os.path.join(dataset_dir, 'maestro-v3.0.0.csv')) as csvfile, open(os.path.join(output_dir, 'train.txt'), 'w') as train_file, open(os.path.join(output_dir, 'validation.txt'), 'w') as validation_file, open(os.path.join(output_dir, 'test.txt'), 'w') as test_file:
            midi_reader = csv.reader(csvfile)
            headers = next(midi_reader)
            header_to_index = {header: index for index, header in enumerate(headers)}

            for midi in midi_reader:
                split = midi[header_to_index['split']]
                midi_filename = midi[header_to_index['midi_filename']]
                midi_path = os.path.join(dataset_dir, midi_filename)
                audio_filename = midi[header_to_index['audio_filename']]
                audio_path = os.path.join(dataset_dir, audio_filename)

                out_line = audio_path + '\n' + midi_path + '\n'
                if split == 'train':
                    utils.write_file_completely(train_file, out_line)
                elif split == 'validation':
                    utils.write_file_completely(validation_file, out_line)
                elif split == 'test':
                    utils.write_file_completely(test_file, out_line)


