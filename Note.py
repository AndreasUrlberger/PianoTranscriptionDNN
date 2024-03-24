from mido import MidiFile, MidiTrack, Message

class Note:
    def __init__(self, value, start_time, end_time):
        self.value = value
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self):
        return f"Note(Value: '{self.value}', Start: '{self.start_time}', End: '{self.end_time}')"

    def midi_to_notes(midi: MidiFile):
        open_notes = {}
        notes = []
        current_time = 0
        # First track contains meta messages, second track contains actual notes.
        for msg in midi.tracks[1]:
            
            # Each note contains the delta time in ticks between the previous note and itself.
            current_time += msg.time
            
            # msg.time is delta time in ticks.
            # msg.note_on with velocity 0 is equivalent to note_off.
            if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in open_notes:
                    start_time = open_notes[msg.note]
                    notes.append(Note(msg.note, start_time, current_time))
                    del open_notes[msg.note]
                else:
                    print(f"Note off without note on. File: {midi.filename}")
            elif msg.type == 'note_on':
                if msg.note in open_notes:
                    print(f"Open note already contained, something must be wrong. File: {midi.filename}")
                open_notes[msg.note] = current_time

        return notes
