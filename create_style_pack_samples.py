import mido
import os

def create_simple_midi(filename, notes, velocity=64, tempo=500000):
    """Create a simple MIDI file with the given notes"""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Add tempo
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    
    # Add time signature
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
    # Add notes
    time = 0
    for note, duration in notes:
        # Note on
        track.append(mido.Message('note_on', note=note, velocity=velocity, time=time))
        time = 0  # Reset time for note off
        
        # Note off
        track.append(mido.Message('note_off', note=note, velocity=0, time=duration))
        
    # Save file
    mid.save(filename)
    print(f"Created MIDI file: {filename}")

# Create directory if it doesn't exist
os.makedirs("style_packs/pop/refs_midi", exist_ok=True)

# Create a simple pop melody
pop_melody = [
    (60, 480),  # C4, quarter note
    (62, 480),  # D4, quarter note
    (64, 480),  # E4, quarter note
    (65, 480),  # F4, quarter note
    (67, 960),  # G4, half note
    (65, 480),  # F4, quarter note
    (64, 480),  # E4, quarter note
    (62, 480),  # D4, quarter note
    (60, 960),  # C4, half note
]

# Create a more complex pop melody
pop_melody2 = [
    (60, 240),  # C4, eighth note
    (60, 240),  # C4, eighth note
    (67, 480),  # G4, quarter note
    (67, 480),  # G4, quarter note
    (69, 240),  # A4, eighth note
    (69, 240),  # A4, eighth note
    (67, 960),  # G4, half note
    (65, 240),  # F4, eighth note
    (65, 240),  # F4, eighth note
    (64, 240),  # E4, eighth note
    (64, 240),  # E4, eighth note
    (62, 480),  # D4, quarter note
    (60, 960),  # C4, half note
]

# Create MIDI files
create_simple_midi("style_packs/pop/refs_midi/pop_melody1.mid", pop_melody)
create_simple_midi("style_packs/pop/refs_midi/pop_melody2.mid", pop_melody2, velocity=80)

# Create a simple audio file placeholder
with open("style_packs/pop/refs_audio/pop_melody1.wav", "w") as f:
    f.write("Placeholder for audio file")

with open("style_packs/pop/refs_audio/pop_melody2.wav", "w") as f:
    f.write("Placeholder for audio file")

print("Created sample files for style pack")
