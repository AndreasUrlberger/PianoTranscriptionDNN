from mido import MidiFile

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

        sustain_pedal_threshold = 10
        sustain_pedal_key = 128
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

            elif msg.type == 'control_change' and msg.control == 64:
                # Sustain pedal.
                if msg.value > sustain_pedal_threshold:
                    # Sustain pedal pressed (note_on).
                    # Only add sustain pedal note if it is not already in open_notes. (This is different from regular notes, for which there is only a single note_on event corresponding to a note_off event.)
                    if sustain_pedal_key not in open_notes:
                        open_notes[sustain_pedal_key] = current_time
                else:
                    # Sustain pedal released (note_off).
                    if sustain_pedal_key in open_notes:
                        start_time = open_notes[sustain_pedal_key]
                        notes.append(Note(sustain_pedal_key, start_time, current_time))
                        del open_notes[sustain_pedal_key]

        return notes
