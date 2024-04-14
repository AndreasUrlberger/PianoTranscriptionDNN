from mido import MidiFile, MidiTrack, Message
from IPython.display import Audio, display
from midi2audio import FluidSynth

def write_test_file():
    # Create a new MIDI file (Type 0)
    mid = MidiFile(type=0)

    # Add a track
    track = MidiTrack()
    mid.tracks.append(track)

    # Add some MIDI messages (e.g., note-on, note-off)
    track.append(Message('note_on', note=60, velocity=64, time=0))
    track.append(Message('note_off', note=60, velocity=64, time=100))

    # Save the MIDI file
    mid.save('output/output2.mid')

def read_midi_file(file_path):
    # Open the MIDI file (clip=True handles notes over 127 velocity)
    mid = MidiFile('output/output.mid', clip=True)

def play_midi(midi_file_path, output_path='output/output.wav'):
    fs = FluidSynth(sound_font='Yamaha_C3_Grand_Piano.sf2')
    fs.midi_to_audio(midi_file=midi_file_path, audio_file=output_path)
    display(Audio(output_path))

def print_midi_info(midi_file_path):
    midi = MidiFile(midi_file_path, clip=True)
    print("Ticks per beat: ", midi.ticks_per_beat)
    print("Type: ", midi.type)
    print("Tracks: ", len(midi.tracks))
    print("-----------------")
    print(midi)
    