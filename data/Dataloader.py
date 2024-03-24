import torch


class MidiDataloader(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split_file):
        self.audio_files = []
        self.midi_files = []
        with open(split_file) as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                audio_file = lines[i].strip()
                midi_file = lines[i+1].strip()
                self.audio_files.append(audio_file)
                self.midi_files.append(midi_file)
        self.dataset_dir = dataset_dir

    def __getitem__(self, index):
        audio_path = os.path.join(self.dataset_dir, self.audio_files[index])
        midi_path = os.path.join(self.dataset_dir, self.midi_files[index])
        audio_tensor, _ = torchaudio.load(audio_path)
        midi_tensor = torch.load(midi_path)
        return audio_tensor, midi_tensor

    def __len__(self):
        return len(self.audio_files)

    def get_audio_path(self, key):
        return self.audio_files[key]

    def get_midi_path(self, key):
        return self.midi_files[key]
