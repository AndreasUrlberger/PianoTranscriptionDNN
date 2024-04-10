"""Data utility functions."""
import os

import torch
import csv

import torchaudio

import Utils as utils
import math
from typing import Callable

from mido import MidiFile
from data.Note import Note
from data.Song import Song


class DatasetUtils:
    def resample_audio(sample_rate_from: int, sample_rate_to: int, audio_tensor: torch.Tensor) -> torch.Tensor:
        resampler = torchaudio.transforms.Resample(
            sample_rate_from, sample_rate_to)
        return resampler(audio_tensor)

    def create_dataset_files(dataset_dir, output_dir):
        with open(os.path.join(dataset_dir, 'maestro-v3.0.0.csv')) as csvfile, open(os.path.join(output_dir, 'train.txt'), 'w') as train_file, open(os.path.join(output_dir, 'validation.txt'), 'w') as validation_file, open(os.path.join(output_dir, 'test.txt'), 'w') as test_file:
            midi_reader = csv.reader(csvfile)
            headers = next(midi_reader)
            header_to_index = {header: index for index,
                               header in enumerate(headers)}

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

    def transform_midi_file(midi_file, discretization) -> torch.Tensor:
        parsed_midi_file = MidiFile(midi_file)
        notes = Note.midi_to_notes(parsed_midi_file)

        tempo = 500000
        for msg in parsed_midi_file.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

        song = Song(notes, parsed_midi_file.length, parsed_midi_file.ticks_per_beat, tempo)
        midi_tensor = song.to_start_time_tensor(discretization_step=discretization)

        return midi_tensor

    def transform_audio_file(audio_file, resample_audio_function: Callable[[int, torch.Tensor], torch.Tensor], midi_chunks, audio_chunk_length) -> torch.Tensor:
        audio_tensor, file_sample_rate = torchaudio.load(audio_file, normalize=True)
        audio_tensor = resample_audio_function(file_sample_rate, audio_tensor)

        # Average the channels to mono
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=False)
        # Rounded up to the nearest multiple of audio_chunk_length
        num_audio_chunks = (audio_tensor.shape[0] + audio_chunk_length - 1) // audio_chunk_length
       
        # Fill up last audio chunk with zeros as padding
        missing_samples = audio_chunk_length - (audio_tensor.shape[0] % audio_chunk_length)
        print('missing samples: ', missing_samples)
        if missing_samples > 0:
            audio_tensor = torch.cat((audio_tensor, torch.zeros(missing_samples)))
        chunked_audio_tensor = audio_tensor.reshape((num_audio_chunks, audio_chunk_length))

        return chunked_audio_tensor
    
    def preprocess_midi_dataset(midi_paths, discretization):
        for index, midi_path in enumerate(midi_paths):
            if index % 100 == 0:
                print(f'Processing midi file {index} of {len(midi_paths)}')
            DatasetUtils._preprocess_midi_file(midi_path, discretization)
    
    def _preprocess_midi_file(midi_path, discretization):
        midi_tensor = DatasetUtils.transform_midi_file(
            midi_path, discretization)
        bare_path, _ = os.path.splitext(midi_path)
        new_path = bare_path + '.pt'
        torch.save(midi_tensor, new_path)
        return midi_tensor


class MidiDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, split, discretization: int, sample_rate: int = 48000):
        assert sample_rate % discretization == 0, 'The sample rate must be divisible by the discretization'
        self.sample_rate = sample_rate
        self.discretization = discretization
        self.root_dir_name = os.path.dirname(dataset_dir)
        file_name = (split + '.txt').lower()
        paths_file = os.path.join(dataset_dir, file_name)
        self.resamplers = {}
        self.waveform_chunk_length = self.sample_rate // self.discretization

        with open(paths_file) as f:
            lines = f.read().splitlines()
            self.audio_files = lines[::2]
            self.midi_files = lines[1::2]

    def __getitem__(self, key):
        audio_file = self.audio_files[key]
        midi_file = self.midi_files[key]

        midi_tensor = DatasetUtils.transform_midi_file(
            midi_file, self.discretization)
        midi_chunk_length = midi_tensor.shape[0]
        audio_tensor = DatasetUtils.transform_audio_file(
            audio_file, self._resample_audio, midi_chunk_length, self.waveform_chunk_length)

        return audio_tensor, midi_tensor

    def get_audio_path(self, key):
        return self.audio_files[key]

    def get_midi_path(self, key):
        return self.midi_files[key]

    def __len__(self):
        # Probably unnecessary
        return len(min(self.audio_files, self.midi_files))

    def _resample_audio(self, file_sample_rate: int, audio_tensor: torch.Tensor) -> torch.Tensor:
        # Resample to self.sample_rate if necessary
        if file_sample_rate != self.sample_rate:
            if file_sample_rate in self.resamplers:
                audio_tensor = self.resamplers[file_sample_rate](audio_tensor)
            else:
                resampler = torchaudio.transforms.Resample(
                    file_sample_rate, self.sample_rate)
                audio_tensor = resampler(audio_tensor)
                self.resamplers[file_sample_rate] = resampler
        return audio_tensor

class MidiIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_dir, split, discretization: int, total_length: int = None, precomputed_midi: bool = False, sample_rate: int = 48000):
        assert sample_rate % discretization == 0, 'The sample rate must be divisible by the discretization'
        self.sample_rate = sample_rate
        self.discretization = discretization
        self.root_dir_name = os.path.dirname(dataset_dir)
        self.precomputed_midi = precomputed_midi
        file_name = (split + '.txt').lower()
        paths_file = os.path.join(dataset_dir, file_name)
        self.resamplers = {}
        self.waveform_chunk_length = self.sample_rate // self.discretization

        with open(paths_file) as f:
            lines = f.read().splitlines()
            self.audio_files = lines[::2]
            self.midi_files = lines[1::2]
        if total_length is None:
            self.length = MidiIterDataset.compute_length(self.midi_files, self.discretization)
        else:
            self.length = total_length

    def _resample_audio(self, file_sample_rate: int, audio_tensor: torch.Tensor) -> torch.Tensor:
        # Resample to self.sample_rate if necessary
        if file_sample_rate != self.sample_rate:
            if file_sample_rate in self.resamplers:
                audio_tensor = self.resamplers[file_sample_rate](audio_tensor)
            else:
                resampler = torchaudio.transforms.Resample(
                    file_sample_rate, self.sample_rate)
                audio_tensor = resampler(audio_tensor)
                self.resamplers[file_sample_rate] = resampler
        return audio_tensor

    def __iter__(self):
        for audio_file, midi_file in zip(self.audio_files, self.midi_files):
            if self.precomputed_midi:
                precomp_path = os.path.splitext(midi_file)[0] + '.pt'
                midi_tensor = torch.load(precomp_path)
            else:
                midi_tensor = DatasetUtils.transform_midi_file(midi_file, self.discretization)
            midi_chunk_length = midi_tensor.shape[0]
            audio_tensor = DatasetUtils.transform_audio_file(audio_file, self._resample_audio, midi_chunk_length, self.waveform_chunk_length)

            for i in range(midi_chunk_length):
                yield audio_tensor[i], midi_tensor[i]

    def __len__(self):
        return self.length

    def compute_length(file_paths, discretization):
        """
        Loops through all files and counts the number of chunks in each file.

        Args:
            file_paths: List of files to loop through
        """
        length = 0
        for file in file_paths:
            midi_file = MidiFile(file)
            midi_frames = math.ceil(midi_file.length * discretization)
            length += midi_frames

        print('computed length: ', length)
        return length
    


class MidiTransformerDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_dir, split, discretization: int, total_length: int = None, precomputed_midi: bool = False, sample_rate: int = 48000, sequence_length: int = 512, collate_fn = None):
        assert sample_rate % discretization == 0, 'The sample rate must be divisible by the discretization'
        self.sample_rate = sample_rate
        self.audio_chunk_length = sample_rate // discretization
        self.discretization = discretization
        self.sequence_length = sequence_length
        self.root_dir_name = os.path.dirname(dataset_dir)
        self.precomputed_midi = precomputed_midi
        file_name = (split + '.txt').lower()
        paths_file = os.path.join(dataset_dir, file_name)
        self.resamplers = {}

        if collate_fn is None:
            self.collate_fn = MidiTransformerDataset._default_collate_fn

        with open(paths_file) as f:
            lines = f.read().splitlines()
            self.audio_files = lines[::2]
            self.midi_files = lines[1::2]

        if total_length is None:
            self.length = MidiIterDataset.compute_length(self.midi_files, self.discretization)
        else:
            self.length = total_length

    def _default_collate_fn(batch, max_length):
        # Shape of batch: (batch_size, sample_length)
        pass



    def _resample_audio(self, file_sample_rate: int, audio_tensor: torch.Tensor) -> torch.Tensor:
        # Resample to self.sample_rate if necessary
        if file_sample_rate != self.sample_rate:
            if file_sample_rate in self.resamplers:
                audio_tensor = self.resamplers[file_sample_rate](audio_tensor)
            else:
                resampler = torchaudio.transforms.Resample(
                    file_sample_rate, self.sample_rate)
                audio_tensor = resampler(audio_tensor)
                self.resamplers[file_sample_rate] = resampler
        return audio_tensor

    def __iter__(self):
        for audio_file, midi_file in zip(self.audio_files, self.midi_files):
            if self.precomputed_midi:
                precomp_path = os.path.splitext(midi_file)[0] + '.pt'
                midi_tensor = torch.load(precomp_path)
            else:
                midi_tensor = DatasetUtils.transform_midi_file(midi_file, self.discretization)
            num_midi_chunks = midi_tensor.shape[0]
            audio_tensor = DatasetUtils.transform_audio_file(audio_file, self._resample_audio, num_midi_chunks, self.audio_chunk_length)
            num_audio_chunks = audio_tensor.shape[0]

            # Pad both tensors to the same number of chunks=-]
            if num_midi_chunks > num_audio_chunks:
                padding = torch.zeros((num_midi_chunks - num_audio_chunks, audio_tensor.shape[1]))
                audio_tensor = torch.cat((audio_tensor, padding))
            elif num_audio_chunks > num_midi_chunks:
                padding = torch.zeros((num_audio_chunks - num_midi_chunks, midi_tensor.shape[1]))
                midi_tensor = torch.cat((midi_tensor, padding))

            # Now pad both tensors to fill up the last sequence
            num_chunks = audio_tensor.shape[0]

            # Return all but the last sequence
            for start_index in range(0, num_chunks - self.sequence_length + 1, self.sequence_length):
                end_index = start_index + self.sequence_length
                audio_sequence = audio_tensor[start_index:end_index, :]
                midi_sequence = midi_tensor[start_index:end_index, :]
                audio_chunks = audio_sequence.reshape(self.sequence_length, -1)
                midi_chunks = midi_sequence.reshape(self.sequence_length, -1)
                padding_mask = torch.ones(self.sequence_length, dtype=torch.bool)
                # TODO How to return the padding mask?
                yield audio_chunks, midi_chunks, padding_mask

            # Last sequence
            length_last_sequence = num_chunks % self.sequence_length
            if length_last_sequence > 0:
                start_index = num_chunks - length_last_sequence
                padding_length = self.sequence_length - length_last_sequence
                audio_sequence = torch.cat([audio_tensor[start_index:, :], torch.zeros((padding_length, audio_tensor.shape[1]))])
                midi_sequence = torch.cat([midi_tensor[start_index:, :], torch.zeros((padding_length, midi_tensor.shape[1]))])
                audio_chunks = audio_sequence.reshape(self.sequence_length, -1)
                midi_chunks = midi_sequence.reshape(self.sequence_length, -1)
                padding_mask = torch.zeros(self.sequence_length, dtype=torch.bool)
                padding_mask[:length_last_sequence] = True
                # TODO How to return the padding mask?
                yield audio_chunks, midi_chunks, padding_mask

    def __len__(self):
        return self.length

    def compute_length(file_paths, discretization):
        """
        Loops through all files and counts the number of chunks in each file.

        Args:
            file_paths: List of files to loop through
        """
        length = 0
        for file in file_paths:
            midi_file = MidiFile(file)
            midi_frames = math.ceil(midi_file.length * discretization)
            length += midi_frames

        print('computed length: ', length)
        return length